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

    # Grid determination parameters
    Nmin: int = 64
    Nmax: int = 2048
    Ninc: int = 32
    n_inner: int = 5
    l_inner: float = 0.1
    f_outer: float = 0.01
    decay_efolds: float = 4.60517
    C: Optional[float] = None
    Cmean: str = 'geometric'

    # Iterative solver parameters
    alpha: Optional[float] = None
    sigma: Optional[Any] = None
    sigma_real_lower: float = 1e-6
    sigma_real_upper: float = 1.0
    sigma_imag_lower: float = -10.0
    sigma_imag_upper: float = 10.0
    sigma_imag: Optional[float] = None
    orderby: str = 'real'
    atol: float = 1e-10
    rtol: float = 1e-5
    gtol: float = 1e-2
    dtol: float = 1e-3
    ntasks: int = 1
    allmodes: bool = False
    suffix: str = ''
    logarithmic: bool = False
    noshear: bool = False
    force: bool = False
    verbose: bool = False
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

    # Optional plot arguments
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    alpha_plot: Optional[float] = None
    value_plot: Optional[float] = None
    file_plot: Optional[str] = None
    dir_plot: Optional[str] = None
    output_plot: Optional[str] = None

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
