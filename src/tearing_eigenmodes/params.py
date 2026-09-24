from dataclasses import dataclass, field
from typing import Any, Optional, List

@dataclass
class SimulationParams:
    # Physical and numerical properties
    eos: str = 'adiabatic'
    parallel_index: float = 3.0
    perpendicular_index: float = 2.0
    CGL: bool = False
    S: Optional[float] = 1e4
    Pr: float = 0.0
    plasma_beta: float = 0.0
    plasma_beta_difference: float = 0.0
    xi: float = 0.0
    Hall: float = 0.0
    a: Optional[float] = 1.0
    w: Optional[float] = 0.0
    zeta: float = 1.0
    delta: Optional[float] = None
    inner_scale: Optional[float] = None
    inner_resolution_safety: float = 1.01

    # Grid determination parameters
    Nmin: int = 64
    Nmax: int = 2048
    Ninc: int = 32
    n_inner: int = 5
    n_equilibrium: int = 5
    n_resistivity: int = 5
    n_inner_scale: int = 5
    n_anisotropy: int = 5
    f_outer: float = 0.01
    decay_efolds: float = 4.60517
    C: Optional[float] = None
    dynamic_C: bool = False

    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.inner_scale is not None and self.delta is None:
            self.delta = self.inner_scale
        elif self.delta is not None and self.inner_scale is None:
            self.inner_scale = self.delta

        if self.n_equilibrium != 5 and self.n_inner == 5:
            self.n_inner = self.n_equilibrium
        elif self.n_inner != 5 and self.n_equilibrium == 5:
            self.n_equilibrium = self.n_inner

        if self.n_inner_scale != 5 and self.n_resistivity == 5:
            self.n_resistivity = self.n_inner_scale
        elif self.n_resistivity != 5 and self.n_inner_scale == 5:
            self.n_inner_scale = self.n_resistivity

        self._initialized = True

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        if not self.__dict__.get('_initialized', False):
            return

        if key == 'inner_scale' and self.__dict__.get('delta') != value:
            super().__setattr__('delta', value)
        elif key == 'delta' and self.__dict__.get('inner_scale') != value:
            super().__setattr__('inner_scale', value)
        elif key == 'n_equilibrium' and self.__dict__.get('n_inner') != value:
            super().__setattr__('n_inner', value)
        elif key == 'n_inner' and self.__dict__.get('n_equilibrium') != value:
            super().__setattr__('n_equilibrium', value)
        elif key == 'n_inner_scale' and self.__dict__.get('n_resistivity') != value:
            super().__setattr__('n_resistivity', value)
        elif key == 'n_resistivity' and self.__dict__.get('n_inner_scale') != value:
            super().__setattr__('n_inner_scale', value)

    # Iterative solver parameters
    alpha: Optional[float] = None
    sigma: Optional[Any] = None
    sigma_real_lower: float = 1e-6
    sigma_real_upper: float = 1.0
    sigma_imag_lower: float = -10.0
    sigma_imag_upper: float = 10.0
    orderby: str = 'real'
    atol: float = 1e-10
    rtol: float = 1e-5
    gtol: float = 1e-2
    # None means 'not chosen'; eigenmodes() then uses 'qz'.
    gevp_method: Optional[str] = 'qz'
    dtol: float = 1e-3
    ntasks: int = 1
    allmodes: bool = False
    suffix: str = ''
    logarithmic: bool = False
    noshear: bool = False
    force: bool = False
    verbose: bool = False
    emit_scale_summary: bool = True
    log_file: Optional[str] = None
    mode: Optional[Any] = None
    dependence: Optional[str] = None
    scan_parameter: Optional[str] = None

    # Optional parameters for range / sweep
    kmin: Optional[float] = None
    kmax: Optional[float] = None
    kinc: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    vinc: Optional[float] = None
    ktol: Optional[float] = None
    kbracket: Optional[List[float]] = None
    step: Optional[float] = None
    extrap_deg: Optional[int] = None
    extrap_guard: Optional[float] = None
    log_extrapolation: bool = False
    step_lower_factor: Optional[float] = None
    step_upper_factor: Optional[float] = None

    # Optional plot arguments
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    alpha_plot: Optional[float] = None
    value_plot: Optional[float] = None
    file_plot: Optional[str] = None
    dir_plot: Optional[str] = None
    output_plot: Optional[str] = None
    nx: int = 100
    nperiods: float = 1.0


    # Path to search or store state files
    data_path: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> List[str]:
        from dataclasses import fields
        return [f.name for f in fields(self)]
