"""System architectures: which phase sequences are physically admissible.

The rest of this package builds phases — boost, parking orbit, deorbit,
midcourse correction, bus dispensing, glide, cruise, ballistic entry. A
*system* is a sequence of them, and not every sequence is a system. This
module states the composition rules, enumerates what satisfies them, and
names the sequences that recur.

Why this is a module and not a table
------------------------------------

It is tempting to hard-code the handful of familiar architectures and move
on. Doing that hides the interesting question, which is *why* the
unfamiliar ones are absent. Some are absent because nobody builds them;
others because they are not physically admissible, and those two look
identical in a table. Encoding the rules and enumerating from them
separates the cases: anything the enumerator produces is admissible, and
anything it refuses comes with the rule that refused it.

The rules are about physics and information, not doctrine:

* **Deorbit presupposes an orbit.** A deorbit burn lowers perigee from a
  parking orbit; without the parking phase there is nothing to lower.
* **Dispensing is exoatmospheric.** A bus separating vehicles inside the
  atmosphere puts them in each other's wake at hypersonic speed. The phase
  must sit before any entry phase.
* **Correction is exoatmospheric.** :mod:`passes.guidance.midcourse`
  solves a Keplerian boundary-value problem; inside the atmosphere the
  plant is no longer Keplerian and the control is aerodynamic, which is
  :mod:`passes.guidance.entry`'s problem instead.
* **Exactly one terminal regime.** A vehicle glides, cruises, or falls
  ballistically. These are not composable in sequence: each is a
  description of the whole atmospheric arc, and chaining two would mean
  the vehicle is in two aerodynamic states at once.
* **The payload must be able to do what the sequence asks, and must use
  what names it.** A single reentry vehicle has nothing to dispense; a
  non-lifting body cannot glide; only a propelled vehicle can cruise. The
  converse matters too: a glide vehicle *can* fly a purely ballistic entry
  by not using its lift, but admitting that would enumerate one physical
  architecture twice under two payload labels, with the second copy
  carrying a description its own trajectory contradicts.

What the enumeration is and is not
----------------------------------

:func:`enumerate_architectures` produces every sequence admissible under
those rules up to a length bound. It is a statement about *composability*,
not about desirability, cost, or whether anyone has built the thing. An
architecture appearing here means the phases chain without contradiction —
nothing more.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "NAMED_ARCHITECTURES",
    "Architecture",
    "Payload",
    "Phase",
    "PhaseRegime",
    "describe",
    "enumerate_architectures",
    "validate",
]


class PhaseRegime(Enum):
    """Where a phase happens, which is what most of the rules turn on."""

    ASCENT = "ascent"
    EXOATMOSPHERIC = "exoatmospheric"
    ATMOSPHERIC = "atmospheric"


class Phase(Enum):
    """One leg of a trajectory."""

    BOOST = "boost"
    PARKING = "parking-orbit"
    DEORBIT = "deorbit"
    MIDCOURSE = "midcourse-correction"
    DISPENSE = "bus-dispensing"
    GLIDE = "glide"
    CRUISE = "powered-cruise"
    BALLISTIC = "ballistic-entry"
    TERMINAL = "terminal-homing"

    @property
    def regime(self) -> PhaseRegime:
        if self is Phase.BOOST:
            return PhaseRegime.ASCENT
        if self in (Phase.PARKING, Phase.DEORBIT, Phase.MIDCOURSE, Phase.DISPENSE):
            return PhaseRegime.EXOATMOSPHERIC
        return PhaseRegime.ATMOSPHERIC


#: The three phases that each describe an entire atmospheric arc. Exactly
#: one must appear, because they are alternative descriptions of the same
#: stretch of flight rather than successive stages of it.
_TERMINAL_REGIMES = (Phase.GLIDE, Phase.CRUISE, Phase.BALLISTIC)


class Payload(Enum):
    """What the boost vehicle is carrying.

    The distinction that matters is capability, not nomenclature: whether
    there is more than one independently targeted body, whether it
    generates useful lift, and whether it carries propulsion for sustained
    flight.
    """

    SINGLE_RV = "single reentry vehicle"
    MULTIPLE_RV = "multiple independently targeted vehicles"
    GLIDER = "hypersonic glide vehicle"
    MULTIPLE_GLIDER = "multiple glide vehicles"
    CRUISER = "powered cruise vehicle"

    @property
    def is_multiple(self) -> bool:
        return self in (Payload.MULTIPLE_RV, Payload.MULTIPLE_GLIDER)

    @property
    def is_lifting(self) -> bool:
        return self in (Payload.GLIDER, Payload.MULTIPLE_GLIDER, Payload.CRUISER)

    @property
    def is_propelled(self) -> bool:
        return self is Payload.CRUISER


@dataclass(frozen=True)
class Architecture:
    """A payload and the ordered phases it flies."""

    phases: tuple[Phase, ...]
    payload: Payload
    name: str = ""

    @property
    def is_orbital(self) -> bool:
        """Whether the profile passes through a parking orbit."""
        return Phase.PARKING in self.phases

    @property
    def terminal_regime(self) -> Phase:
        for phase in self.phases:
            if phase in _TERMINAL_REGIMES:
                return phase
        raise ValueError("architecture has no terminal regime")

    def __str__(self) -> str:
        return " -> ".join(p.value for p in self.phases)


def validate(phases: tuple[Phase, ...] | list[Phase], payload: Payload) -> None:
    """Raise :class:`ValueError` if the sequence is not admissible.

    Each rule reports which rule failed and why, rather than a generic
    rejection: the point of encoding them is to distinguish "not built" from
    "not possible", and a bare boolean loses that.
    """
    seq = tuple(phases)
    if not seq:
        raise ValueError("an architecture needs at least one phase")
    if len(set(seq)) != len(seq):
        raise ValueError(f"phases must not repeat, got {[p.value for p in seq]}")
    if seq[0] is not Phase.BOOST:
        raise ValueError(
            f"every architecture begins with boost; this one begins with {seq[0].value}"
        )

    terminals = [p for p in seq if p in _TERMINAL_REGIMES]
    if len(terminals) != 1:
        raise ValueError(
            f"exactly one of glide, cruise or ballistic entry must appear — "
            f"they are alternative descriptions of the whole atmospheric arc, "
            f"not successive stages. Got {[p.value for p in terminals]}"
        )
    terminal = terminals[0]

    if Phase.DEORBIT in seq and Phase.PARKING not in seq:
        raise ValueError(
            "deorbit presupposes an orbit: there is no perigee to lower without a parking phase"
        )
    if Phase.PARKING in seq and Phase.DEORBIT not in seq:
        raise ValueError(
            "a parking orbit must be left by a deorbit burn; without one the "
            "vehicle stays in orbit and there is no entry"
        )
    if Phase.PARKING in seq and seq.index(Phase.PARKING) > seq.index(Phase.DEORBIT):
        raise ValueError("deorbit cannot precede the parking orbit it leaves")

    # Every exoatmospheric phase must precede the atmospheric arc.
    entry_index = seq.index(terminal)
    for phase in seq:
        if phase.regime is PhaseRegime.EXOATMOSPHERIC and seq.index(phase) > entry_index:
            raise ValueError(f"{phase.value} is exoatmospheric and cannot follow {terminal.value}")
    if Phase.TERMINAL in seq and seq.index(Phase.TERMINAL) < entry_index:
        raise ValueError("terminal homing is the last phase, not the first")

    # Payload capability.
    if Phase.DISPENSE in seq and not payload.is_multiple:
        raise ValueError(
            f"a {payload.value} has nothing to dispense; bus dispensing "
            f"requires more than one independently targeted body"
        )
    if payload.is_multiple and Phase.DISPENSE not in seq:
        raise ValueError(
            f"a {payload.value} must dispense them; without that phase the "
            f"bodies never separate and the payload is effectively single"
        )
    # The terminal regime is fixed by the payload, in both directions.
    #
    # One direction is capability: a non-lifting body cannot glide and an
    # unpropelled one cannot cruise. The other is less obvious and is what
    # keeps the enumeration honest. A glide vehicle *can* fly a purely
    # ballistic entry by simply not using its lift, and a cruiser can glide
    # with the engine off. Admitting those would enumerate the same
    # physical architecture twice under two payload labels, and the second
    # copy carries a payload description that the trajectory contradicts.
    # An architecture that does not use a capability should be described
    # with the payload that lacks it.
    expected = {
        Payload.SINGLE_RV: Phase.BALLISTIC,
        Payload.MULTIPLE_RV: Phase.BALLISTIC,
        Payload.GLIDER: Phase.GLIDE,
        Payload.MULTIPLE_GLIDER: Phase.GLIDE,
        Payload.CRUISER: Phase.CRUISE,
    }[payload]
    if terminal is not expected:
        if terminal is Phase.GLIDE and not payload.is_lifting:
            raise ValueError(f"a {payload.value} generates no useful lift and cannot glide")
        if terminal is Phase.CRUISE and not payload.is_propelled:
            raise ValueError(f"a {payload.value} carries no propulsion and cannot cruise")
        raise ValueError(
            f"a {payload.value} flying {terminal.value} does not use the "
            f"capability that names it; describe this architecture with a "
            f"payload whose terminal regime is {terminal.value} rather than "
            f"enumerating the same trajectory twice"
        )
    if payload is Payload.CRUISER and Phase.PARKING in seq:
        raise ValueError(
            "an air-breathing cruiser cannot be staged through a parking "
            "orbit: it needs atmosphere to operate and orbit is where there "
            "is none"
        )


def enumerate_architectures(payload: Payload | None = None) -> list[Architecture]:
    """All admissible phase sequences, optionally for one payload.

    Enumerates every subset of the optional phases in every order, keeps
    those :func:`validate` accepts, and returns them sorted by length then
    by phase order. The search is exhaustive over the phase set, which is
    small enough that this costs nothing and removes any question of a
    combination having been overlooked.
    """
    payloads = list(Payload) if payload is None else [payload]
    optional = [p for p in Phase if p is not Phase.BOOST]
    found: list[Architecture] = []
    for candidate_payload in payloads:
        for count in range(1, len(optional) + 1):
            for subset in itertools.combinations(optional, count):
                for order in itertools.permutations(subset):
                    sequence = (Phase.BOOST, *order)
                    try:
                        validate(sequence, candidate_payload)
                    except ValueError:
                        continue
                    found.append(Architecture(phases=sequence, payload=candidate_payload))
    found.sort(key=lambda a: (len(a.phases), [p.value for p in a.phases]))
    return found


def describe(architecture: Architecture) -> str:
    """One-line human summary of what an architecture is."""
    regime = architecture.terminal_regime
    orbital = "fractional-orbital" if architecture.is_orbital else "suborbital"
    multiplicity = "multiple bodies" if architecture.payload.is_multiple else "single body"
    return f"{orbital}, {regime.value}, {multiplicity} ({len(architecture.phases)} phases)"


def _named(name: str, payload: Payload, *phases: Phase) -> Architecture:
    validate(phases, payload)
    return Architecture(phases=phases, payload=payload, name=name)


#: The architectures that recur, each validated at import so a typo here
#: fails loudly rather than shipping an inadmissible reference case.
NAMED_ARCHITECTURES: dict[str, Architecture] = {
    a.name: a
    for a in (
        _named(
            "ballistic-single",
            Payload.SINGLE_RV,
            Phase.BOOST,
            Phase.MIDCOURSE,
            Phase.BALLISTIC,
            Phase.TERMINAL,
        ),
        _named(
            "ballistic-multiple",
            Payload.MULTIPLE_RV,
            Phase.BOOST,
            Phase.MIDCOURSE,
            Phase.DISPENSE,
            Phase.BALLISTIC,
            Phase.TERMINAL,
        ),
        _named(
            "boost-glide",
            Payload.GLIDER,
            Phase.BOOST,
            Phase.GLIDE,
            Phase.TERMINAL,
        ),
        _named(
            "boost-glide-multiple",
            Payload.MULTIPLE_GLIDER,
            Phase.BOOST,
            Phase.MIDCOURSE,
            Phase.DISPENSE,
            Phase.GLIDE,
            Phase.TERMINAL,
        ),
        _named(
            "fractional-orbital-single",
            Payload.SINGLE_RV,
            Phase.BOOST,
            Phase.PARKING,
            Phase.DEORBIT,
            Phase.BALLISTIC,
            Phase.TERMINAL,
        ),
        _named(
            "fractional-orbital-glide",
            Payload.GLIDER,
            Phase.BOOST,
            Phase.PARKING,
            Phase.DEORBIT,
            Phase.GLIDE,
            Phase.TERMINAL,
        ),
        _named(
            "fractional-orbital-multiple",
            Payload.MULTIPLE_RV,
            Phase.BOOST,
            Phase.PARKING,
            Phase.DEORBIT,
            Phase.DISPENSE,
            Phase.BALLISTIC,
            Phase.TERMINAL,
        ),
        _named(
            "fractional-orbital-multiple-glide",
            Payload.MULTIPLE_GLIDER,
            Phase.BOOST,
            Phase.PARKING,
            Phase.DEORBIT,
            Phase.DISPENSE,
            Phase.GLIDE,
            Phase.TERMINAL,
        ),
        _named(
            "powered-cruise",
            Payload.CRUISER,
            Phase.BOOST,
            Phase.CRUISE,
            Phase.TERMINAL,
        ),
    )
}
