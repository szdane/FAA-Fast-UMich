"""
Modular PVS proof builder for collision avoidance certificates.

This module provides a clean, flexible interface for generating PVS proof scripts
that can handle arbitrary numbers of piecewise domains.

Architecture:
- Interval: Represents domain types (left_open, right_open, ci)
- Domain: A domain with its interval, trajectory function, and associated lemma
- ProofNode: Building block for constructing proof trees
- CaseSplitter: Generates nested CASE structures with proper parenthesization
- ProofBuilder: Main class for constructing complete proofs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum, auto
from sympy import symbols, diff, sympify


# =============================================================================
# INTERVAL TYPES
# =============================================================================


class IntervalType(Enum):
    """Types of intervals supported in PVS proofs."""
    LEFT_OPEN = auto()      # (-inf, bound] : x <= bound
    RIGHT_OPEN = auto()     # [bound, inf)  : x >= bound
    CLOSED = auto()         # [low, high]   : low <= x <= high
    ALL_REALS = auto()      # (-inf, inf)   : all real numbers


@dataclass
class Interval:
    """
    Represents a mathematical interval.

    Examples:
        Interval(IntervalType.LEFT_OPEN, high=0)     -> (-inf, 0]
        Interval(IntervalType.RIGHT_OPEN, low=5)     -> [5, inf)
        Interval(IntervalType.CLOSED, low=0, high=5) -> [0, 5]
    """
    interval_type: IntervalType
    low: Optional[str] = None
    high: Optional[str] = None

    def to_pvs(self) -> str:
        """Convert to PVS domain notation."""
        if self.interval_type == IntervalType.LEFT_OPEN:
            return f"left_open({self.high})"
        elif self.interval_type == IntervalType.RIGHT_OPEN:
            return f"right_open({self.low})"
        elif self.interval_type == IntervalType.CLOSED:
            return f"ci({self.low}, {self.high})"
        else:
            return "real"

    def pvs_type_name(self) -> str:
        """Get just the type name without parameters."""
        if self.interval_type == IntervalType.LEFT_OPEN:
            return "left_open"
        elif self.interval_type == IntervalType.RIGHT_OPEN:
            return "right_open"
        elif self.interval_type == IntervalType.CLOSED:
            return "ci"
        else:
            return "real"

    def bound_params(self) -> List[str]:
        """Get the bound parameters for this interval."""
        if self.interval_type == IntervalType.LEFT_OPEN:
            return [self.high]
        elif self.interval_type == IntervalType.RIGHT_OPEN:
            return [self.low]
        elif self.interval_type == IntervalType.CLOSED:
            return [self.low, self.high]
        else:
            return []

    def case_condition(self) -> Optional[str]:
        """
        Get the CASE condition for this interval boundary.

        Returns the condition that determines if x is in this interval
        rather than subsequent ones. Used for nested CASE statements.
        """
        if self.interval_type == IntervalType.LEFT_OPEN:
            return f"x <= {self.high}"
        elif self.interval_type == IntervalType.CLOSED:
            # For closed intervals, we use the upper bound for the CASE split
            return f"x <= {self.high}"
        elif self.interval_type == IntervalType.RIGHT_OPEN:
            # Right-open intervals are typically last, no CASE needed
            return None
        else:
            return None


# =============================================================================
# DOMAIN REPRESENTATION
# =============================================================================


@dataclass
class Domain:
    """
    Represents a domain segment with its interval, trajectory, and lemma.

    Attributes:
        interval: The interval type and bounds
        trajectory: The clipped trajectory function as a PVS expression (for helper proof)
                   This is the COND-based clipped function, e.g.,
                   "LAMBDA(x:real): COND x <= -1 -> g(x), ELSE -> g(-1) ENDCOND"
        lemma_name: Name of the per-segment lemma (e.g., "le_lo_case_0")
        simplified_trajectory: The simplified trajectory expression valid in this domain
                              (for main proof). This is the actual expression without COND,
                              e.g., "x^2 + 2*x + 1" for a polynomial domain.
                              If None, defaults to trajectory.
        deriv_direction: "ge" for >= 0, "le" for <= 0, or both
        deriv_bounds: List of derivative bounds (e.g., ["0"] or ["0", "8"])
    """
    interval: Interval
    trajectory: str
    lemma_name: str
    case_index: int = 0
    simplified_trajectory: Optional[str] = None  # For main proof CASE statements
    deriv_direction: str = "le"  # "le" or "ge" or "both"
    deriv_bounds: List[str] = field(default_factory=lambda: ["0"])


# =============================================================================
# PROOF NODE - BUILDING BLOCK FOR PROOF TREES
# =============================================================================


class ProofNode:
    """
    A node in the proof tree.

    Can be a tactic call (THEN, SPREAD, CASE, etc.) with children,
    or a leaf tactic (PROPAX, ASSERT, etc.).

    Handles parenthesization automatically when converting to string.
    """

    def __init__(self, tactic: str, children: Optional[List['ProofNode']] = None, arg: str = ""):
        """
        Create a proof node.

        Args:
            tactic: The tactic name (e.g., "THEN", "SPREAD", "CASE", "PROPAX")
            children: List of child proof nodes
            arg: Argument to the tactic (e.g., "x <= 0" for CASE)
        """
        self.tactic = tactic
        self.children = children or []
        self.arg = arg

    def to_pvs(self, indent: int = 0) -> str:
        """
        Convert to PVS proof script format.

        Args:
            indent: Current indentation level

        Returns:
            Formatted PVS proof string with proper parenthesization
        """
        if not self.children:
            # Leaf node
            if self.arg:
                return f"({self.tactic} {self.arg})"
            else:
                return f"({self.tactic})"

        # Node with children
        if self.arg:
            header = f"({self.tactic} {self.arg}"
        else:
            header = f"({self.tactic}"

        if self.tactic == "SPREAD":
            # SPREAD has special formatting: first child is the tactic,
            # rest are in a list
            if len(self.children) >= 2:
                first_child = self.children[0].to_pvs(indent + 1)
                rest = [c.to_pvs(indent + 2) for c in self.children[1:]]
                return f"{header} {first_child}\n{'%|-' + ' ' * (indent + 1)}({' '.join(rest)}))"
            elif len(self.children) == 1:
                return f"{header} {self.children[0].to_pvs(indent + 1)})"
            else:
                return f"{header})"

        elif self.tactic == "THEN":
            # THEN chains tactics together
            parts = [c.to_pvs(indent + 1) for c in self.children]
            return f"{header} {' '.join(parts)})"

        else:
            # Generic node with children
            parts = [c.to_pvs(indent + 1) for c in self.children]
            return f"{header} {' '.join(parts)})"


# Convenience functions for creating proof nodes
def THEN(*children) -> ProofNode:
    return ProofNode("THEN", list(children))

def SPREAD(tactic: ProofNode, *branches) -> ProofNode:
    return ProofNode("SPREAD", [tactic] + list(branches))

def CASE(condition: str) -> ProofNode:
    return ProofNode("CASE", arg=f'"{condition}"')

def SPLIT(arg: str = "-1") -> ProofNode:
    return ProofNode("SPLIT", arg=arg)

def PROPAX() -> ProofNode:
    return ProofNode("PROPAX")

def ASSERT() -> ProofNode:
    return ProofNode("ASSERT")

def SKEEP() -> ProofNode:
    return ProofNode("SKEEP")

def SKOLETIN(arg: str = "*") -> ProofNode:
    return ProofNode("SKOLETIN", arg=arg)

def FLATTEN() -> ProofNode:
    return ProofNode("FLATTEN")

def LEMMA(name: str) -> ProofNode:
    return ProofNode("LEMMA", arg=f'"{name}"')

def INST(formula_num: str, *args) -> ProofNode:
    args_str = " ".join(f'"{a}"' for a in args)
    return ProofNode("INST", arg=f'{formula_num} {args_str}')

def EXPAND(name: str, num: str = "") -> ProofNode:
    if num:
        return ProofNode("EXPAND", arg=f'"{name}" {num}')
    else:
        return ProofNode("EXPAND", arg=f'"{name}"')

def TYPEPRED(*vars) -> ProofNode:
    args_str = " ".join(f'"{v}"' for v in vars)
    return ProofNode("TYPEPRED", arg=args_str)

def GRIND() -> ProofNode:
    return ProofNode("GRIND")

def HIDE_ALL_BUT(*nums) -> ProofNode:
    if len(nums) == 1:
        return ProofNode("HIDE-ALL-BUT", arg=str(nums[0]))
    else:
        return ProofNode("HIDE-ALL-BUT", arg=f"({' '.join(str(n) for n in nums)})")

def DECOMPOSE_EQUALITY(num: int) -> ProofNode:
    return ProofNode("DECOMPOSE-EQUALITY", arg=str(num))

def DERIVABLE() -> ProofNode:
    return ProofNode("DERIVABLE")

def DERIV() -> ProofNode:
    return ProofNode("DERIV")

def HIDE(num: int) -> ProofNode:
    return ProofNode("HIDE", arg=str(num))

def REPLACE(num: int) -> ProofNode:
    return ProofNode("REPLACE", arg=str(num))


# =============================================================================
# DERIVATIVE ANALYSIS HELPERS
# =============================================================================


def _is_constant_derivative(simplified_expr_str: str) -> bool:
    """
    Check if the derivative of an expression w.r.t. x is constant.

    This is used to determine the correct proof tactics for domain branches.
    Quadratic functions (x^2) have non-constant derivatives (2x), while
    linear functions (-10x - 25) have constant derivatives (-10).

    Args:
        simplified_expr_str: The simplified trajectory expression as a string,
                            e.g., "x^2 + 2*x + 1" or "-10*x - 25"

    Returns:
        True if the derivative is constant (doesn't depend on x),
        False otherwise.
    """
    try:
        x = symbols('x')
        # Convert PVS-style exponentiation to Python style
        expr_str = simplified_expr_str.replace('^', '**')
        expr = sympify(expr_str)
        deriv = diff(expr, x)
        # Check if derivative contains x (non-constant) or not (constant)
        return not deriv.has(x)
    except Exception:
        # If parsing fails, assume non-constant (safer - uses SPREAD pattern)
        return False


# =============================================================================
# PROOF LINE FORMATTER
# =============================================================================


class ProofFormatter:
    """
    Formats proof scripts with proper PVS prooflite syntax.

    Handles:
    - Adding %|- prefixes to each line
    - Managing line wrapping for long lines
    - Proper indentation
    """

    def __init__(self, max_line_length: int = 100):
        self.max_line_length = max_line_length
        self.lines: List[str] = []

    def add_header(self, lemma_name: str):
        """Add the proof header."""
        self.lines.append(f"%|- {lemma_name} : PROOF")

    def add_line(self, content: str, indent: int = 0):
        """Add a proof line with proper prefix and indentation."""
        prefix = "%|- " + " " * indent
        self.lines.append(f"{prefix}{content}")

    def add_footer(self, lemma_name: str):
        """Add the proof footer."""
        self.lines.append(f"%|- QED {lemma_name}")

    def to_string(self) -> str:
        """Get the complete proof as a string."""
        return "\n".join(self.lines)


# =============================================================================
# CASE SPLITTER - GENERATES NESTED CASE STRUCTURES
# =============================================================================


class CaseSplitter:
    """
    Generates nested CASE structures for domain boundaries.

    For domains with boundaries [b1, b2, ..., b(n-1)]:
    - First domain: CASE "x <= b1"
    - Second domain (in first's else): CASE "x <= b2"
    - ... and so on until last domain falls through to else
    """

    @staticmethod
    def generate_case_chain(
        boundaries: List[str],
        case_proofs: List[str],
    ) -> str:
        """
        Generate a chain of nested CASE statements.

        Args:
            boundaries: List of boundary values for CASE splits
            case_proofs: List of proof strings for each case
                        (should have len(boundaries) + 1 elements for complete coverage)

        Returns:
            Nested CASE structure as a string

        Example:
            boundaries = ["-1", "3"]
            case_proofs = [proof_0, proof_1, proof_2]

            Generates:
            (SPREAD (CASE "x <= -1")
             ((proof_0)
              (SPREAD (CASE "x <= 3")
               ((proof_1)
                (proof_2)))))
        """
        if len(boundaries) == 0:
            # Single case, no splitting needed
            return case_proofs[0] if case_proofs else ""

        if len(case_proofs) != len(boundaries) + 1:
            raise ValueError(
                f"Expected {len(boundaries) + 1} case proofs for {len(boundaries)} boundaries, "
                f"got {len(case_proofs)}"
            )

        # Build from inside out (last case first)
        result = case_proofs[-1]

        # Work backwards through boundaries
        for i in range(len(boundaries) - 1, -1, -1):
            boundary = boundaries[i]
            true_branch = case_proofs[i]
            false_branch = result

            result = f"""(SPREAD (CASE "x <= {boundary}")
 (({true_branch})
  {false_branch}))"""

        return result


# =============================================================================
# PER-DOMAIN PROOF TEMPLATES
# =============================================================================


class DomainProofTemplate:
    """
    Templates for generating proofs for each domain type.

    Different interval types require different proof tactics:
    - left_open: Uses mvt_gen_le_lo or mvt_gen_ge_lo
    - right_open: Uses mvt_gen_ge_ro or mvt_gen_le_ro
    - ci (closed): Uses mvt_gen_ge_ci or mvt_gen_le_ci

    Position matters:
    - First domain: Might have special handling
    - Middle domains: Standard handling
    - Last domain: Falls through to else in CASE chain
    """

    @staticmethod
    def get_mvt_lemma(interval: Interval, direction: str) -> str:
        """
        Get the appropriate MVT lemma name.

        Args:
            interval: The interval type
            direction: "le" for <= or "ge" for >=

        Returns:
            Name of the MVT lemma to use
        """
        type_name = interval.pvs_type_name()
        if type_name == "left_open":
            return f"mvt_gen_{direction}_lo"
        elif type_name == "right_open":
            return f"mvt_gen_{direction}_ro"
        elif type_name == "ci":
            return f"mvt_gen_{direction}_ci"
        else:
            return f"mvt_gen_{direction}_real"

    @staticmethod
    def generate_per_segment_proof(
        domain: Domain,
        position: str,  # "first", "middle", or "last"
        polygon_half_width: str = "2",
    ) -> str:
        """
        Generate proof for a single domain segment (per-segment lemma proof).

        This is the proof inside the CASE statement for one domain.
        Structure depends on interval type and position.
        """
        interval = domain.interval
        direction = domain.deriv_direction
        mvt_lemma = DomainProofTemplate.get_mvt_lemma(interval, direction)

        # Get the bound(s) for the interval
        bounds = interval.bound_params()
        type_name = interval.pvs_type_name()

        # Common variables
        deriv_bound = domain.deriv_bounds[0] if domain.deriv_bounds else "0"

        if interval.interval_type == IntervalType.LEFT_OPEN:
            bound = bounds[0]
            return DomainProofTemplate._left_open_proof(
                mvt_lemma, bound, deriv_bound, type_name, polygon_half_width
            )
        elif interval.interval_type == IntervalType.RIGHT_OPEN:
            bound = bounds[0]
            return DomainProofTemplate._right_open_proof(
                mvt_lemma, bound, deriv_bound, type_name, polygon_half_width
            )
        elif interval.interval_type == IntervalType.CLOSED:
            return DomainProofTemplate._closed_interval_proof(
                mvt_lemma, bounds[0], bounds[1], deriv_bound, type_name, polygon_half_width
            )
        else:
            return "% Unsupported interval type"

    @staticmethod
    def _left_open_proof(
        mvt_lemma: str,
        bound: str,
        deriv_bound: str,
        type_name: str,
        half_width: str,
    ) -> str:
        """Generate proof for left_open interval (-inf, bound]."""
        return f"""(THEN (SKEEP*) (SKOLETIN*) (FLATTEN) (SKEEP)
  (SPREAD (CASE "xo + {half_width} <= {bound}")
   ((THEN (LEMMA "{mvt_lemma}")
     (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "xo + {half_width}" "x")
      ((SPREAD (SPLIT -1)
        ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "x" "xo - {half_width}")
           ((ASSERT) (THEN (EXPAND "{type_name}") (ASSERT))
            (THEN (EXPAND "{type_name}") (ASSERT)))))
         (PROPAX) (PROPAX) (PROPAX)))
       (THEN (EXPAND "{type_name}") (ASSERT))
       (THEN (EXPAND "{type_name}") (ASSERT)))))
    (THEN (LEMMA "{mvt_lemma}")
     (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "{bound}" "x")
      ((SPREAD (SPLIT -1)
        ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "x" "xo - {half_width}")
           ((THEN (EXPAND "f") (EXPAND "{type_name}") (SPREAD (SPLIT -1) ((ASSERT) (PROPAX))))
            (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))
            (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT)))))
         (THEN (ASSERT) (PROPAX)) (THEN (ASSERT) (PROPAX)) (THEN (ASSERT) (PROPAX))))
       (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))
       (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))))))))"""

    @staticmethod
    def _right_open_proof(
        mvt_lemma: str,
        bound: str,
        deriv_bound: str,
        type_name: str,
        half_width: str,
    ) -> str:
        """Generate proof for right_open interval [bound, inf)."""
        return f"""(THEN (SKEEP*) (SKOLETIN*) (FLATTEN) (SKEEP)
  (SPREAD (CASE "xo - {half_width} >= {bound}")
   ((THEN (LEMMA "{mvt_lemma}")
     (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "xo + {half_width}" "x")
      ((SPREAD (SPLIT -1)
        ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "x" "xo - {half_width}")
           ((ASSERT) (THEN (EXPAND "{type_name}") (ASSERT))
            (THEN (EXPAND "{type_name}") (ASSERT)))))
         (PROPAX) (PROPAX) (PROPAX)))
       (THEN (EXPAND "{type_name}") (ASSERT))
       (THEN (EXPAND "{type_name}") (ASSERT)))))
    (THEN (LEMMA "{mvt_lemma}")
     (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "xo + {half_width}" "x")
      ((SPREAD (SPLIT -1)
        ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{bound}" "{deriv_bound}" "x" "{bound}")
           ((THEN (EXPAND "f") (EXPAND "{type_name}") (SPREAD (SPLIT -1) ((ASSERT) (PROPAX))))
            (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))
            (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT)))))
         (THEN (ASSERT) (PROPAX)) (THEN (ASSERT) (PROPAX)) (THEN (ASSERT) (PROPAX))))
       (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))
       (THEN (EXPAND "f") (EXPAND "{type_name}") (ASSERT))))))))"""

    @staticmethod
    def _closed_interval_proof(
        mvt_lemma: str,
        low: str,
        high: str,
        deriv_bound: str,
        type_name: str,
        half_width: str,
    ) -> str:
        """Generate proof for closed interval [low, high]."""
        return f"""(THEN (SKEEP*) (SKOLETIN*) (FLATTEN) (SKEEP)
  (SPREAD (CASE "xo - {half_width} >= {low} AND xo + {half_width} <= {high}")
   ((THEN (FLATTEN) (LEMMA "{mvt_lemma}")
     (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "xo + {half_width}" "x")
      ((SPREAD (SPLIT -1)
        ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "x" "xo - {half_width}")
           ((ASSERT) (THEN (EXPAND "{type_name}") (PROPAX))
            (THEN (ASSERT) (EXPAND "{type_name}") (PROPAX)))))
         (ASSERT) (PROPAX) (PROPAX) (PROPAX)))
       (THEN (EXPAND "{type_name}") (ASSERT)) (THEN (EXPAND "{type_name}") (ASSERT)))))
    (SPREAD (CASE "xo - {half_width} < {low} AND xo + {half_width} <= {high}")
     ((THEN (FLATTEN) (LEMMA "{mvt_lemma}")
       (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "xo + {half_width}" "x")
        ((SPREAD (SPLIT -1)
          ((THEN (ASSERT) (LEMMA "{mvt_lemma}")
            (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "x" "{low}")
             ((THEN (EXPAND "f") (ASSERT)) (THEN (EXPAND "{type_name}") (PROPAX))
              (THEN (ASSERT) (EXPAND "{type_name}") (PROPAX)))))
           (ASSERT) (PROPAX) (PROPAX) (PROPAX)))
         (THEN (EXPAND "{type_name}") (ASSERT)) (THEN (EXPAND "{type_name}") (ASSERT)))))
      (THEN (ASSERT)
       (SPREAD (CASE "xo - {half_width} >= {low} AND xo + {half_width} > {high}")
        ((THEN (FLATTEN) (LEMMA "{mvt_lemma}")
          (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "{high}" "x")
           ((SPREAD (SPLIT -1)
             ((THEN (LEMMA "{mvt_lemma}")
               (SPREAD (INST -1 "f" "{low}" "{high}" "{deriv_bound}" "x" "xo - {half_width}")
                ((SPREAD (SPLIT -1)
                  ((THEN (ASSERT) (EXPAND "f") (ASSERT)) (ASSERT) (PROPAX)
                   (ASSERT) (PROPAX)))
                 (THEN (EXPAND "{type_name}") (ASSERT))
                 (THEN (EXPAND "{type_name}") (PROPAX)))))
              (ASSERT) (PROPAX) (ASSERT) (PROPAX)))
            (THEN (EXPAND "{type_name}") (PROPAX))
            (THEN (ASSERT) (EXPAND "{type_name}") (PROPAX)))))
         (THEN (ASSERT)
          (SPREAD (CASE "xo - {half_width} < {low} AND xo + {half_width} > {high}") ((GRIND) (GRIND))))))))))))"""


# =============================================================================
# UNIFYING HELPER PROOF BUILDER
# =============================================================================


class UnifyingHelperProofBuilder:
    """
    Builds the full_domain_soundness_lemma_helper proof.

    This proof ties together the per-segment lemmas using nested CASE statements.
    """

    @staticmethod
    def build(
        domains: List[Domain],
        boundaries: List[str],
    ) -> str:
        """
        Generate the unifying helper lemma proof.

        Args:
            domains: List of Domain objects for each piecewise segment
            boundaries: List of boundary values for CASE splits

        Returns:
            Complete PVS proof script string
        """
        if len(domains) == 0:
            return "% No domains provided"

        if len(domains) == 1:
            # Single domain - simpler proof
            return UnifyingHelperProofBuilder._single_domain_proof(domains[0])

        # Generate proof for each domain
        case_proofs = []
        for i, domain in enumerate(domains):
            position = "first" if i == 0 else ("last" if i == len(domains) - 1 else "middle")
            case_proof = UnifyingHelperProofBuilder._generate_case_proof(domain, position, i)
            case_proofs.append(case_proof)

        # Build nested CASE structure
        nested_cases = UnifyingHelperProofBuilder._build_nested_cases(
            boundaries, case_proofs, domains
        )

        return f"""%|- full_domain_soundness_lemma_helper : PROOF
%|- (THEN (SKEEP) (SKOLETIN*) (FLATTEN)
{nested_cases}
%|- QED full_domain_soundness_lemma_helper
"""

    @staticmethod
    def _single_domain_proof(domain: Domain) -> str:
        """Generate proof for single domain case."""
        return f"""%|- full_domain_soundness_lemma_helper : PROOF
%|- (THEN (SKEEP) (SKOLETIN*) (FLATTEN)
%|-  (LEMMA "{domain.lemma_name}") (INST -1 "xo" "yo" "g") (ASSERT))
%|- QED full_domain_soundness_lemma_helper
"""

    @staticmethod
    def _generate_case_proof(domain: Domain, position: str, case_index: int) -> str:
        """
        Generate the proof content for a single case in the helper.

        Different from per-segment lemma proofs - this invokes the per-segment
        lemma and handles the trajectory equality assertion.
        """
        lemma_name = domain.lemma_name
        interval = domain.interval
        trajectory = domain.trajectory
        expand_fn = f"f{case_index}"

        # Build the trajectory equality CASE
        domain_pvs = interval.to_pvs()
        traj_eq = f'restrict[real, ({domain_pvs}), real](g) = (restrict[real, ({domain_pvs}), real]({trajectory}))'

        # Number of PROPAX/ASSERT depends on position
        if position == "first":
            split_tactics = "(PROPAX) (PROPAX) (PROPAX) (PROPAX) (PROPAX) (ASSERT) (ASSERT)"
            inst_tactic = "(THEN (INST 1 \"x\") (ASSERT))"
        elif position == "last":
            split_tactics = "(ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT)"
            inst_tactic = "(THEN (ASSERT) (INST 1 \"x\") (ASSERT))"
        else:  # middle
            # For middle (ci) domains: 8 conditions before EXISTS
            # derivable? + deriv conditions from all domains that precede/include this one
            split_tactics = "(ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT) (ASSERT)"
            inst_tactic = "(THEN (INST 1 \"x\") (ASSERT))"

        # Build the proof structure
        if position == "first":
            return f"""(THEN (LEMMA "{lemma_name}") (INST -1 "xo" "yo" "g") (ASSERT) (EXPAND "{expand_fn}")
%|-     (ASSERT)
%|-     (SPREAD
%|-      (CASE
%|-          "{traj_eq}")
%|-      ((SPREAD (SPLIT -2)
%|-        ({split_tactics}
%|-         {inst_tactic}))
%|-       (THEN (DECOMPOSE-EQUALITY 1) (HIDE-ALL-BUT 1) (TYPEPRED "x!1")
%|-        (GRIND)))))"""
        elif position == "last":
            return f"""(THEN (LEMMA "{lemma_name}") (INST -1 "xo" "yo" "g") (ASSERT)
%|-       (EXPAND "{expand_fn}")
%|-       (SPREAD
%|-        (CASE
%|-            "{traj_eq}")
%|-        ((SPREAD (SPLIT -2)
%|-          ({split_tactics}
%|-           {inst_tactic}))
%|-         (THEN (HIDE-ALL-BUT 1) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!1")
%|-          (GRIND)))))"""
        else:  # middle
            return f"""(THEN (LEMMA "{lemma_name}") (INST -1 "xo" "yo" "g") (ASSERT)
%|-       (EXPAND "{expand_fn}")
%|-       (SPREAD
%|-        (CASE
%|-            "{traj_eq}")
%|-        ((SPREAD (SPLIT -2)
%|-          ({split_tactics}
%|-           {inst_tactic}))
%|-         (THEN (ASSERT) (DECOMPOSE-EQUALITY 1) (HIDE-ALL-BUT 1) (TYPEPRED "x!1")
%|-          (GRIND))
%|-         (THEN (ASSERT) (HIDE-ALL-BUT 1) (GRIND)))))"""

    @staticmethod
    def _build_nested_cases(
        boundaries: List[str],
        case_proofs: List[str],
        domains: List[Domain],
    ) -> str:
        """
        Build the nested CASE structure for the helper proof.

        Structure for 3 cases with boundaries [b1, b2]:
        (SPREAD (CASE "x <= b1")
         ((case_0_proof)
          (SPREAD (CASE "x <= b2")
           ((case_1_proof)
            (case_2_proof)))))

        Parenthesis count for 3-case:
        - Outer SPREAD needs 2 closes at the very end
        - First case list needs 1 close after case_0
        - Inner SPREAD needs 2 closes
        - Inner case list needs 1 close after case_1 and 1 after case_2
        """
        if len(boundaries) == 0:
            return f"%|-  {case_proofs[0]})"

        if len(boundaries) == 1:
            # 2-case: simple structure
            # case_proofs[i] already starts with (THEN
            return f"""%|-  (SPREAD (CASE "x <= {boundaries[0]}")
%|-   ({case_proofs[0]}
%|-    {case_proofs[1]})))"""

        if len(boundaries) == 2:
            # 3-case: one level of nesting
            # Structure exactly matches example_6.pvs:
            # (SPREAD (CASE "x <= b1")
            #  ((case_0_proof)   <-- one ( for list, case_0 already starts with (THEN
            #   (SPREAD (CASE "x <= b2")
            #    ((case_1_proof)  <-- one ( for list
            #     (case_2_proof)))))  <-- closes: case_2, inner list, inner SPREAD list, outer SPREAD, THEN = 6
            # case_proofs[i] already starts with (THEN and ends with appropriate closes
            return f"""%|-  (SPREAD (CASE "x <= {boundaries[0]}")
%|-   ({case_proofs[0]}
%|-    (SPREAD (CASE "x <= {boundaries[1]}")
%|-     ({case_proofs[1]}
%|-      {case_proofs[2]}))))))"""

        # 4+ cases: build iteratively
        lines = []

        # First case opening
        lines.append(f'%|-  (SPREAD (CASE "x <= {boundaries[0]}")')
        lines.append(f"%|-   ({case_proofs[0]}")

        # Middle cases - each opens a new SPREAD/CASE
        for i in range(1, len(boundaries)):
            lines.append(f'%|-    (SPREAD (CASE "x <= {boundaries[i]}")')
            lines.append(f"%|-     ({case_proofs[i]}")

        # Last case - no leading ( since it's a continuation of the list
        lines.append(f"%|-      {case_proofs[-1]}")

        # Close: 2 for each SPREAD, 1 for the outer list
        # For n boundaries: n SPREADs each need )), plus outer ) = 2n + 1
        closes = ")" * (2 * len(boundaries) + 1)
        lines[-1] += closes

        return "\n".join(lines)


# =============================================================================
# MAIN UNIFYING PROOF BUILDER
# =============================================================================


class UnifyingProofBuilder:
    """
    Builds the full_domain_soundness_lemma proof (main unifying proof).

    This proof uses the helper lemma and handles DERIVABLE/DERIV branches
    for each domain with proper trajectory equality assertions.
    """

    @staticmethod
    def build(
        domains: List[Domain],
        full_trajectory: str,
        piecewise_boundaries: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the main unifying lemma proof.

        Args:
            domains: List of Domain objects
            full_trajectory: The full piecewise trajectory expression
            piecewise_boundaries: Optional boundaries where trajectory changes
                                 (for special CASE handling)

        Returns:
            Complete PVS proof script string
        """
        num_cases = len(domains)

        if num_cases == 0:
            return "% No domains provided"

        if num_cases == 1:
            return UnifyingProofBuilder._single_domain_proof(domains[0])

        # Calculate PROPAX count
        # Pattern: 10 for 2 cases, 16 for 3 cases, 6*n - 2 for n >= 3 cases
        if num_cases == 2:
            num_propax = 10
        elif num_cases >= 3:
            num_propax = 6 * num_cases - 2

        # Format PROPAX calls
        propax_lines = UnifyingProofBuilder._format_propax(num_propax)

        # Generate domain branches (DERIVABLE and DERIV for each)
        domain_branches = []
        for i, domain in enumerate(domains):
            branches = UnifyingProofBuilder._generate_domain_branches(
                domain, i, len(domains), full_trajectory, piecewise_boundaries
            )
            domain_branches.extend(branches)

        # Build the complete proof
        return UnifyingProofBuilder._assemble_proof(
            propax_lines, domain_branches, num_cases
        )

    @staticmethod
    def _single_domain_proof(domain: Domain) -> str:
        """Generate proof for single domain."""
        return f"""%|- full_domain_soundness_lemma : PROOF
%|- (THEN (SKEEP) (SKOLETIN*) (FLATTEN) (EXPAND "g_1") (LEMMA "{domain.lemma_name}")
%|-  (ASSERT))
%|- QED full_domain_soundness_lemma
"""

    @staticmethod
    def _format_propax(count: int) -> str:
        """Format PROPAX calls into lines of 8."""
        lines = []
        remaining = count
        while remaining > 0:
            batch = min(8, remaining)
            lines.append(" ".join(["(PROPAX)"] * batch))
            remaining -= batch
        return "\n%|-    ".join(lines)

    @staticmethod
    def _generate_domain_branches(
        domain: Domain,
        index: int,
        total: int,
        full_trajectory: str,
        piecewise_boundaries: Optional[List[str]],
    ) -> List[str]:
        """
        Generate DERIVABLE and DERIV branches for a domain.

        Returns list of proof branch strings.
        """
        interval = domain.interval
        domain_pvs = interval.to_pvs()
        type_name = interval.pvs_type_name()
        bounds = interval.bound_params()

        # Get the simplified trajectory for this domain
        # This should be the actual expression (not the clipped COND version)
        simplified = domain.simplified_trajectory
        if simplified is None:
            # Fallback: try to extract from trajectory or use full_trajectory
            simplified = full_trajectory

        # Build trajectory equality for CASE
        # Left side: full trajectory (with COND) restricted to domain
        # Right side: simplified expression valid in this domain
        # Replace x with s for the lambda
        traj_eq = (
            f"(LAMBDA(s: ({domain_pvs})): {full_trajectory.replace('x', 's')}) = "
            f"(LAMBDA (s: ({domain_pvs})): {simplified.replace('x', 's')})"
        )

        is_last = (index == total - 1)
        # For the last domain, check if there's any piecewise boundary
        # (the boundary where the trajectory function changes expression)
        has_piecewise = is_last and piecewise_boundaries and len(piecewise_boundaries) > 0

        if index == 0:
            # First domain - pass simplified expression for derivative detection
            return UnifyingProofBuilder._first_domain_branches(
                domain_pvs, type_name, bounds, traj_eq, simplified
            )
        elif is_last:
            # Last domain - may have piecewise handling
            if has_piecewise:
                return UnifyingProofBuilder._last_domain_piecewise_branches(
                    domain_pvs, type_name, bounds, traj_eq, piecewise_boundaries[-1]
                )
            else:
                return UnifyingProofBuilder._last_domain_branches(
                    domain_pvs, type_name, bounds, traj_eq
                )
        else:
            # Middle domain
            return UnifyingProofBuilder._middle_domain_branches(
                domain_pvs, type_name, bounds, traj_eq
            )

    @staticmethod
    def _first_domain_branches(
        domain_pvs: str, type_name: str, bounds: List[str], traj_eq: str,
        simplified_expr: Optional[str] = None
    ) -> List[str]:
        """
        Generate branches for first domain.

        Args:
            domain_pvs: The PVS domain string, e.g., "left_open(-5)"
            type_name: The interval type name, e.g., "left_open"
            bounds: The boundary values
            traj_eq: The trajectory equation string
            simplified_expr: The simplified trajectory expression for this domain,
                           used to detect if derivative is constant
        """
        bound = bounds[0] if bounds else ""
        # traj_eq is like "(LAMBDA...full_piecewise...) = (LAMBDA...simplified...)"
        # We need to change it to "restrict[real, (domain), real](g_1) = (LAMBDA...simplified...)"
        # Extract the simplified RHS from traj_eq
        rhs = traj_eq.split(" = ")[1] if " = " in traj_eq else traj_eq
        restrict_eq = f"restrict[real, ({domain_pvs}), real](g_1) = {rhs}"

        derivable = f"""(THEN (HIDE-ALL-BUT 1)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (HIDE -1) (DERIVABLE) (LEMMA "{type_name}_dd")
%|-        (INST -1 "{bound}"))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!1")
%|-        (EXPAND "g_1") (GRIND)))))"""

        # Check if the derivative is constant (linear first piece)
        # vs non-constant (quadratic/polynomial first piece)
        is_constant = False
        if simplified_expr:
            is_constant = _is_constant_derivative(simplified_expr)

        if is_constant:
            # For constant derivative (linear functions like -6*x - 72),
            # DERIV produces a single sequent with both deriv_domain? and the
            # inequality in the succedent (not 2 separate subgoals).
            # Use sequential steps: DERIV, then LEMMA to discharge deriv_domain?,
            # then prove the inequality.
            deriv = f"""(THEN (HIDE-ALL-BUT 1) (SKEEP)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (DERIV) (LEMMA "{type_name}_dd") (INST -1 "{bound}")
%|-        (TYPEPRED "x!1") (EXPAND "{type_name}" -1) (ASSERT))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!2")
%|-        (EXPAND "g_1") (GRIND)))))"""
        else:
            # For non-constant derivative (quadratic/polynomial functions),
            # DERIV produces 2 subgoals, so use SPREAD pattern
            deriv = f"""(THEN (HIDE-ALL-BUT 1) (SKEEP)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1)
%|-        (SPREAD (DERIV)
%|-         ((THEN (TYPEPRED "x!1") (EXPAND "{type_name}" -1) (ASSERT))
%|-          (THEN (LEMMA "{type_name}_dd") (INST -1 "{bound}")))))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!2")
%|-        (EXPAND "g_1") (GRIND)))))"""

        return [derivable, deriv]

    @staticmethod
    def _middle_domain_branches(
        domain_pvs: str, type_name: str, bounds: List[str], traj_eq: str
    ) -> List[str]:
        """Generate branches for middle domain."""
        # Extract the simplified RHS from traj_eq
        rhs = traj_eq.split(" = ")[1] if " = " in traj_eq else traj_eq
        restrict_eq = f"restrict[real, ({domain_pvs}), real](g_1) = {rhs}"

        derivable = f"""(THEN (HIDE-ALL-BUT 1)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (DERIVABLE 1))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!1")
%|-        (EXPAND "g_1") (GRIND)))))"""

        # For middle domains (ci), DERIV produces one subgoal, not multiple
        # So we use DERIV followed by tactics directly, not SPREAD
        deriv1 = f"""(THEN (HIDE-ALL-BUT 1) (SKEEP)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (HIDE -1) (DERIV) (TYPEPRED "x!1") (EXPAND "{type_name}" -1)
%|-        (PROPAX))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!2")
%|-        (EXPAND "g_1") (GRIND)))))"""

        return [derivable, deriv1]

    @staticmethod
    def _last_domain_branches(
        domain_pvs: str, type_name: str, bounds: List[str], traj_eq: str
    ) -> List[str]:
        """Generate branches for last domain without piecewise handling."""
        # Extract the simplified RHS from traj_eq
        rhs = traj_eq.split(" = ")[1] if " = " in traj_eq else traj_eq
        restrict_eq = f"restrict[real, ({domain_pvs}), real](g_1) = {rhs}"

        derivable = f"""(THEN (HIDE-ALL-BUT 1)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (DERIVABLE))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!1")
%|-        (EXPAND "g_1") (GRIND)))))"""

        # For last domain (right_open), DERIV produces one subgoal
        # So we use DERIV followed by nothing else (or just HIDE -1)
        deriv = f"""(THEN (HIDE-ALL-BUT 1) (SKEEP)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (HIDE -1) (DERIV))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!2")
%|-        (EXPAND "g_1") (GRIND)))))"""

        return [derivable, deriv]

    @staticmethod
    def _last_domain_piecewise_branches(
        domain_pvs: str, type_name: str, bounds: List[str], traj_eq: str, split_val: str
    ) -> List[str]:
        """Generate branches for last domain with piecewise handling."""
        # Extract the simplified RHS from traj_eq
        rhs = traj_eq.split(" = ")[1] if " = " in traj_eq else traj_eq
        restrict_eq = f"restrict[real, ({domain_pvs}), real](g_1) = {rhs}"

        derivable = f"""(THEN (HIDE-ALL-BUT 1)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (DERIVABLE))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!1")
%|-        (EXPAND "g_1") (GRIND)))))"""

        # For last domain (right_open), DERIV produces one subgoal
        deriv = f"""(THEN (HIDE-ALL-BUT 1) (SKEEP)
%|-     (SPREAD
%|-      (CASE "{restrict_eq}")
%|-      ((THEN (REPLACE -1) (HIDE -1) (DERIV))
%|-       (THEN (HIDE 2) (DECOMPOSE-EQUALITY 1) (TYPEPRED "x!2")
%|-        (EXPAND "g_1") (GRIND)))))"""

        return [derivable, deriv]

    @staticmethod
    def _assemble_proof(
        propax_lines: str,
        domain_branches: List[str],
        num_cases: int,
    ) -> str:
        """Assemble the complete proof."""
        # Join branches with proper formatting
        branches_str = "\n%|-    ".join(domain_branches)

        # Trailing PROPAX for 3+ cases
        # Note: trailing includes one extra ) to close the outer THEN
        trailing = ""
        if num_cases >= 3:
            trailing = "\n%|-    " + " ".join(["(PROPAX)"] * 6) + ")"

        # Use SKOLETIN* (no space) for all cases, or SKOLETIN 1 for specific
        # For 2 cases, use (SKOLETIN*), for 3+ use (SKOLETIN 1)
        skoletin_tactic = "(SKOLETIN*)" if num_cases < 3 else "(SKOLETIN 1)"

        # Closing parens: )) closes branch-list and SPREAD
        # For 2 cases, we need an extra ) to close the outer THEN
        # For 3+ cases, the trailing already has one )
        close_parens = ")))" if num_cases < 3 else "))"

        return f"""%|- full_domain_soundness_lemma : PROOF
%|- (THEN (SKEEP) {skoletin_tactic} (FLATTEN) (LEMMA "full_domain_soundness_lemma_helper")
%|-  (INST -1 "x" "xo" "yo" "g_1")
%|-  (SPREAD (SPLIT -1)
%|-   ({propax_lines}
%|-    {branches_str}{trailing}{close_parens}
%|- QED full_domain_soundness_lemma"""


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================


def create_left_open_domain(
    bound: str,
    trajectory: str,
    lemma_name: str,
    case_index: int = 0,
    direction: str = "le",
    deriv_bounds: List[str] = None,
    simplified_trajectory: Optional[str] = None,
) -> Domain:
    """Create a left-open domain (-inf, bound]."""
    return Domain(
        interval=Interval(IntervalType.LEFT_OPEN, high=bound),
        trajectory=trajectory,
        lemma_name=lemma_name,
        case_index=case_index,
        simplified_trajectory=simplified_trajectory,
        deriv_direction=direction,
        deriv_bounds=deriv_bounds or ["0"],
    )


def create_right_open_domain(
    bound: str,
    trajectory: str,
    lemma_name: str,
    case_index: int = 0,
    direction: str = "ge",
    deriv_bounds: List[str] = None,
    simplified_trajectory: Optional[str] = None,
) -> Domain:
    """Create a right-open domain [bound, inf)."""
    return Domain(
        interval=Interval(IntervalType.RIGHT_OPEN, low=bound),
        trajectory=trajectory,
        lemma_name=lemma_name,
        case_index=case_index,
        simplified_trajectory=simplified_trajectory,
        deriv_direction=direction,
        deriv_bounds=deriv_bounds or ["0"],
    )


def create_closed_domain(
    low: str,
    high: str,
    trajectory: str,
    lemma_name: str,
    case_index: int = 0,
    direction: str = "ge",
    deriv_bounds: List[str] = None,
    simplified_trajectory: Optional[str] = None,
) -> Domain:
    """Create a closed interval domain [low, high]."""
    return Domain(
        interval=Interval(IntervalType.CLOSED, low=low, high=high),
        trajectory=trajectory,
        lemma_name=lemma_name,
        case_index=case_index,
        simplified_trajectory=simplified_trajectory,
        deriv_direction=direction,
        deriv_bounds=deriv_bounds or ["0"],
    )


# =============================================================================
# MAIN API
# =============================================================================


def generate_unifying_proofs(
    domains: List[Domain],
    boundaries: List[str],
    full_trajectory: str,
    piecewise_boundaries: Optional[List[str]] = None,
) -> tuple[str, str]:
    """
    Generate both the helper and main unifying proofs.

    Args:
        domains: List of Domain objects for each piecewise segment
        boundaries: List of boundary values for domain splits
        full_trajectory: The full piecewise trajectory expression
        piecewise_boundaries: Optional boundaries for piecewise CASE handling

    Returns:
        Tuple of (helper_proof, main_proof)
    """
    helper_proof = UnifyingHelperProofBuilder.build(domains, boundaries)
    main_proof = UnifyingProofBuilder.build(domains, full_trajectory, piecewise_boundaries)

    return helper_proof, main_proof
