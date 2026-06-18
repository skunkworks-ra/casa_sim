from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class ConfigError(Exception):
    """Structured config validation error."""

    def __init__(self, rule: str, field_path: str, message: str):
        self.rule = rule
        self.field_path = field_path
        self.message = message
        super().__init__(f"[{rule}] {field_path}: {message}")


@dataclass
class ObsposConfig:
    mode: str                          # known | geodetic | itrf
    value: Any                         # str | dict


@dataclass
class AntennaEntry:
    name: str
    x: float
    y: float
    z: float
    diameter: float


@dataclass
class CannedObservatory:
    telescope: str
    cfg_file: Optional[str] = None
    antlist: Optional[List[str]] = None


@dataclass
class CustomObservatory:
    telname: str
    mount: str
    obspos: ObsposConfig
    antennas: List[AntennaEntry] = field(default_factory=list)


@dataclass
class ObservatoryConfig:
    mode: str                          # canned | custom
    feeds: str                         # e.g. "perfect R L"
    canned: Optional[CannedObservatory] = None
    custom: Optional[CustomObservatory] = None


@dataclass
class FieldConfig:
    name: str
    direction: str                     # e.g. "J2000 19h59m28.5s +40d40m00.0s"


@dataclass
class SpwConfig:
    name: str
    freq: str                          # e.g. "1.0GHz"
    deltafreq: str                     # e.g. "0.2GHz"
    nchan: int
    stokes: str                        # e.g. "RR LL"
    freqresolution: Optional[str] = None   # None → resolved in observation.py


@dataclass
class ObserveCall:
    field: str
    spw: str
    start_time: str
    stop_time: str


@dataclass
class ObservationConfig:
    epoch: str
    integration_time: str
    use_hourangle: bool
    fields: List[FieldConfig]
    spws: List[SpwConfig]
    observe_calls: List[ObserveCall]


@dataclass
class CLStokesSpectrum:
    component_index: int
    type: str                          # spectral_index | tabular
    ref_freq: Optional[str] = None
    index: Optional[List[float]] = None   # [alpha_I, alpha_fraclin, alpha_pa, alpha_fraccir]
    freqs: Optional[List[float]] = None
    I: Optional[List[float]] = None
    Q: Optional[List[float]] = None
    U: Optional[List[float]] = None
    V: Optional[List[float]] = None


@dataclass
class SourceDef:
    """Inline source definition for building a component list from YAML."""
    name: str
    direction: str                     # e.g. "J2000 13h31m08.29s +30d30m32.96s"
    flux: List[float]                  # [I] or [I,Q,U,V]
    ref_freq: str                      # e.g. "1.4GHz"
    spectral_index: List[float] = field(default_factory=lambda: [0.0])  # [alpha] or [alpha, beta]
    shape: str = 'point'               # point | gaussian | disk
    major: Optional[str] = None        # e.g. "10arcsec" (gaussian/disk)
    minor: Optional[str] = None
    pa: Optional[str] = None           # position angle
    rm: float = 0.0                    # rotation measure rad/m^2
    frac_pol: Optional[float] = None   # fractional linear polarization (derives Q,U from I)
    chi: Optional[float] = None        # EVPA in degrees (used with frac_pol)


@dataclass
class FaradayConfig:
    enabled: bool
    rm_mode: str                       # global | map
    rm_value: Union[float, str]        # scalar rad/m^2 or image path
    ref_freq: str


@dataclass
class SpectralLineConfig:
    name: str
    stokes: str
    channels: List[int]
    flux_profile: List[float]
    mode: str                          # point | gaussian | image
    direction: Optional[str] = None
    major: Optional[str] = None
    minor: Optional[str] = None
    pa: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class TRecsSpectralConfig:
    mode: str                          # trecs_sed | synthetic
    ref_freq: str                      # e.g. "1.4GHz"
    spidx_dist: Optional[Dict[str, Any]] = None  # synthetic mode only


@dataclass
class TRecsPolarizationConfig:
    pol_fraction_source: str           # trecs | synthetic
    pol_fraction_dist: Optional[Dict[str, Any]] = None
    pol_spidx_dist: Optional[Dict[str, Any]] = None
    rm_dist: Optional[Dict[str, Any]] = None
    chi0_dist: Optional[Dict[str, Any]] = None
    n_rm_components: Union[int, Dict[str, Any]] = 1


@dataclass
class TRecsConfig:
    catalog_paths: Dict[str, str]      # keys: agn and/or sfg; values: file paths
    flux_floor_jy: float
    flux_floor_col: str                # column name, e.g. "I1400"
    field_size_arcsec: float
    tile_center_deg: List[float]       # [x_center, y_center] in tile Euclidean coords
    seed: int
    spectral: TRecsSpectralConfig
    polarization: TRecsPolarizationConfig
    readme_path: Optional[str] = None


@dataclass
class CorpusMorphologyConfig:
    """
    Controls the extended-morphology component of a corpus_mix sky model.

    field_type:
        One of "point_only", "diffuse_dominant", "central_shell_flow",
        "fully_diffuse", or null.  When null, a type is sampled per-field
        from the balance weights each time the sky model is built.

    balance:
        Optional override for the sampling balance.  Dict from field-type
        name (string) to relative weight (float).  All four types must be
        present if this key is given.  Weights are normalised internally.
        If null, the module-level DEFAULT_FIELD_TYPE_BALANCE is used:
            point_only 0.25 / diffuse_dominant 0.32 /
            central_shell_flow 0.33 / fully_diffuse 0.10.

    seed:
        RNG seed for morphology sampling.  Independent of trecs.seed so
        the two populations can be varied separately.
        Defaults to 0.

    repo_root:
        Filesystem path to the casa_sim repo root (containing
        data/morphology/lib/).  If null, auto-detected from the module
        location (correct for in-tree installs).
    """
    field_type: Optional[str] = None   # null → sampled from balance
    balance: Optional[Dict[str, float]] = None
    seed: int = 0
    repo_root: Optional[str] = None


@dataclass
class SkyModelConfig:
    stokes: str                        # I | IQUV
    mode: str                          # component_list | image_native | image_extrapolate | t_recs | corpus_mix
    cl_path: Optional[str] = None
    sources: Optional[List[SourceDef]] = None   # inline source definitions → auto-builds .cl
    cl_stokes_spectrum: Optional[List[CLStokesSpectrum]] = None
    image_path: Optional[str] = None
    ref_freq: Optional[str] = None     # image_extrapolate
    alpha_mode: Optional[str] = None   # global | map
    alpha_value: Optional[Union[float, str]] = None
    faraday: Optional[FaradayConfig] = None
    spectral_lines: List[SpectralLineConfig] = field(default_factory=list)
    trecs: Optional[TRecsConfig] = None
    corpus_morphology: Optional[CorpusMorphologyConfig] = None


@dataclass
class PredictionConfig:
    gridder: str                       # standard | mosaic | awproject
    predictor: str                     # auto | ft_dft | sm_predict | tclean
    normtype: str                      # flatsky | flatnoise
    cell: Optional[str] = None        # None → derived
    imsize: Optional[int] = None      # None → derived


@dataclass
class NoiseConfig:
    enabled: bool
    mode: str                          # simplenoise | tsys-atm | per_baseline
    value: Optional[str] = None        # Jy string; simplenoise only


@dataclass
class GainsConfig:
    enabled: bool
    mode: str                          # fbm
    amplitude: float = 0.0


@dataclass
class CorruptionConfig:
    seed: Optional[int]                # None → non-reproducible
    noise: NoiseConfig = field(default_factory=lambda: NoiseConfig(enabled=False, mode='simplenoise'))
    gains: GainsConfig = field(default_factory=lambda: GainsConfig(enabled=False, mode='fbm'))


@dataclass
class ImagingConfig:
    enabled: bool
    deconvolver: str = 'hogbom'
    nterms: Optional[int] = None
    pbcor: bool = False
    niter: int = 500
    export_fits: bool = False   # write dirty.fits and psf.fits alongside the MS
    pblimit: float = 0.05       # primary beam mask threshold; set negative to disable
    specmode: Optional[str] = None   # override tclean specmode; None = auto (mfs if 1 chan else cube)
    stokes: Optional[str] = None     # override tclean stokes; None = use sky_model.stokes
    imsize: Optional[int] = None     # imaging grid size; None = effective_imsize. Set larger than
                                     # the model/predict imsize for a guard band: sources land in the
                                     # central region, aliased sidelobes do not wrap. Dirty/PSF come
                                     # out at this size (no crop); model stays at effective_imsize.


@dataclass
class SweepAxis:
    parameter: str                     # dot-path e.g. "corruption.noise.value"
    values: List[Any]


@dataclass
class SweepConfig:
    axes: List[SweepAxis]


@dataclass
class SimConfig:
    """Top-level config. Carries derived imaging params after derive_imaging_params()."""
    name: str
    observatory: ObservatoryConfig
    observation: ObservationConfig
    sky_model: SkyModelConfig
    prediction: PredictionConfig
    corruption: CorruptionConfig
    imaging: ImagingConfig
    sweep: Optional[SweepConfig] = None

    # Derived in derive_imaging_params() — not in YAML
    _derived_cell: Optional[str] = field(default=None, repr=False)
    _derived_imsize: Optional[int] = field(default=None, repr=False)

    @property
    def effective_cell(self) -> Optional[str]:
        return self.prediction.cell or self._derived_cell

    @property
    def effective_imsize(self) -> Optional[int]:
        return self.prediction.imsize or self._derived_imsize
