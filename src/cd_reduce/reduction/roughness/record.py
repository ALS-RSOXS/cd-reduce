"""Self-describing PSD result records (CSV with a JSON metadata header)."""

from __future__ import annotations

import dataclasses
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cd_reduce.reduction.roughness.cross_section import (
    CrossSection,
    DevelopedLine,
    LatentFilm,
)
from cd_reduce.reduction.roughness.extract import PsdTable

CONVENTION = "# two-sided PSD, nm^3, f = Qy/2pi in nm^-1"
_CLASSES = {"LatentFilm": LatentFilm, "DevelopedLine": DevelopedLine}
_OPT = ("S_cross", "var_cross")
_VEC = ("f_center", "S_lpr", "S_lwr", "S_ler", "var_lpr", "var_lwr", "var_ler",
        "cov_pw", "cond", "chi2_red", "n")  # fmt: skip


def _enc(o: Any) -> Any:
    if isinstance(o, complex):
        return {"__complex__": [o.real, o.imag]}
    if isinstance(o, np.generic):
        return _enc(o.item())
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _dec(d: dict) -> Any:
    return complex(*d["__complex__"]) if "__complex__" in d else d


def cross_section_dict(xs: CrossSection) -> dict:
    """``{"class": name, "params": dataclasses.asdict(xs)}``."""
    return {"class": type(xs).__name__, "params": dataclasses.asdict(xs)}  # type: ignore[call-overload]


@dataclass
class PsdRecord:
    """A PSD table plus everything needed to interpret and rescale it.

    ``cross_section`` is ``{"class", "params"}``. On ``load`` it stays a dict
    (it is never converted back to an object); use :meth:`cross_section_obj`
    to rebuild LatentFilm/DevelopedLine, which returns ``None`` for unknown names.
    """

    table: PsdTable
    cross_section: dict
    scale: float
    scale_err: float
    scale_source: Literal["bragg_orders", "known"]
    source: dict
    window: tuple[float, float]
    sigma_win: dict[str, tuple[float, float]]
    model_fit: dict | None = None

    def cross_section_obj(self) -> CrossSection | None:
        cls = _CLASSES.get(self.cross_section.get("class", ""))
        return cls(**self.cross_section["params"]) if cls else None

    def _meta(self) -> dict:
        return {
            "cross_section": self.cross_section, "scale": self.scale,
            "scale_err": self.scale_err, "scale_source": self.scale_source,
            "source": self.source, "window": list(self.window),
            "sigma_win": {k: list(v) for k, v in self.sigma_win.items()},
            "model_fit": self.model_fit,
        }  # fmt: skip

    def save(self, path: str | Path) -> None:
        t = self.table
        cols: dict[str, Any] = {"f_lo": t.f_edges[:-1], "f_hi": t.f_edges[1:]}
        for n in (*_VEC, *_OPT):
            v = getattr(t, n)
            if v is not None:
                cols[n] = v
        buf = io.StringIO()
        pd.DataFrame(cols).to_csv(buf, index=False)
        meta = json.dumps(self._meta(), default=_enc)
        Path(path).write_text(f"{CONVENTION}\n# {meta}\n{buf.getvalue()}")

    @classmethod
    def load(cls, path: str | Path) -> PsdRecord:
        lines = Path(path).read_text().splitlines()
        if not lines or lines[0] != CONVENTION:
            raise ValueError("not a PSD record: missing convention line")
        meta = json.loads(lines[1][2:], object_hook=_dec)
        df = pd.read_csv(io.StringIO("\n".join(lines[2:])))
        arr = {n: df[n].to_numpy() for n in df.columns}
        edges = np.append(arr["f_lo"], arr["f_hi"][-1])
        kw: dict[str, Any] = {n: arr[n].astype(float) for n in _VEC if n != "n"}
        kw["n"] = arr["n"].astype(np.int64)
        for n in _OPT:
            kw[n] = arr[n].astype(float) if n in arr else None
        return cls(
            table=PsdTable(f_edges=edges, **kw),
            cross_section=meta["cross_section"],
            scale=meta["scale"], scale_err=meta["scale_err"],
            scale_source=meta["scale_source"], source=meta["source"],
            window=tuple(meta["window"]),
            sigma_win={k: tuple(v) for k, v in meta["sigma_win"].items()},
            model_fit=meta["model_fit"],
        )  # fmt: skip

    def rescale(
        self, new_xs: CrossSection, *, energy_ev: float | None = None
    ) -> PsdRecord:
        """Rescale to a new cross section: S scales by the ratio, var by ratio**2.

        The ratio is the contrast ``|dn_A|^2 |Gamma|^2`` old/new. ``Gamma`` is
        the unit-contrast form factor, so ``|dn_A|^2`` is included (doubling
        ``|dn_A|`` divides S by 4). ``cov_pw`` also scales by ratio**2 and
        ``sigma_win`` (value and error) by sqrt(ratio). ``model_fit`` is
        dropped (set to ``None``): it was fit to the old S.

        Note: exact rescaling needs the per-pixel Qz distribution. Here the
        ratio is evaluated at ``source["qz_mean"]`` (0.0 if absent), an
        approximation. The old cross-section is rebuilt from the record, so
        it must be LatentFilm or DevelopedLine.

        Rescaling applies to one energy only. If ``source`` lists more than
        one energy, ``energy_ev`` must name the one whose cross section is
        replaced (its entry in ``source["cross_section_by_energy"]`` is the old
        cross section); otherwise ``ValueError`` is raised.
        """
        energies = list(self.source.get("energies_ev", []))
        by_e = self.source.get("cross_section_by_energy")
        if energy_ev is None and len(energies) > 1:
            raise ValueError(
                f"record has energies {energies}; rescale applies to one energy, "
                "pass energy_ev= to pick it"
            )
        src = self.source
        if energy_ev is not None and by_e:
            j = int(np.argmin([abs(c["energy_ev"] - energy_ev) for c in by_e]))
            if abs(by_e[j]["energy_ev"] - energy_ev) > 1e-6:
                raise ValueError(f"no cross section for energy {energy_ev} eV")
            entry = by_e[j]["cross_section"]
            cls = _CLASSES.get(entry.get("class", ""))
            old = cls(**entry["params"]) if cls else None
            new_list = [dict(c) for c in by_e]
            new_list[j]["cross_section"] = cross_section_dict(new_xs)
            src = {**self.source, "cross_section_by_energy": new_list}
        else:
            old = self.cross_section_obj()
        if old is None:
            raise ValueError("cannot rescale: unknown cross-section class")
        qz = float(self.source.get("qz_mean", 0.0))
        ratio = float(
            abs(old.dn_A) ** 2
            * old.gamma2(qz)
            / (abs(new_xs.dn_A) ** 2 * new_xs.gamma2(qz))
        )
        t = self.table
        upd = {}
        for n in ("S_lpr", "S_lwr", "S_ler", "S_cross"):
            if getattr(t, n) is not None:
                upd[n] = getattr(t, n) * ratio
        for n in ("var_lpr", "var_lwr", "var_ler", "var_cross", "cov_pw"):
            if getattr(t, n) is not None:
                upd[n] = getattr(t, n) * ratio**2
        sw = {
            k: (v[0] * np.sqrt(ratio), v[1] * np.sqrt(ratio))
            for k, v in self.sigma_win.items()
        }
        lowest = energy_ev is None or not by_e or j == 0
        return replace(
            self, table=replace(t, **upd),
            cross_section=cross_section_dict(new_xs) if lowest else self.cross_section,
            sigma_win=sw, source=src, model_fit=None,
        )  # fmt: skip
