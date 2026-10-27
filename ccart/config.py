"""Static configuration for ConceptCartographer.

Holds stop words, generic-word lists, the relation lexicon used to classify
predicates (Predicate-Argument Structure, Unit IV) and colour palettes.
"""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down during
    each either else few for from further had has have having he her here hers herself him
    himself his how however i if in into is it its itself just let may me might more most
    must my myself neither no nor not now of off on once one only or other ought our ours
    ourselves out over own per same shall she should so some such than that the their
    theirs them themselves then there these they this those through thus to too under until
    up upon us very was we were what when where whether which while who whom whose why will
    with within without would yet you your yours yourself yourselves s t d ll m re ve y
    et al eg ie etc vs""".split()
)

# Words that are rarely a meaningful stand-alone "concept" in a textbook.
GENERIC_SINGLE = frozenset(
    """example figure fig chapter section way thing lot kind type fact case number sense
    process technique method approach idea task step item list part form place point
    term time end set value use work today result problem line side group level order
    answer question reader book page text-book textbook unit exercise summary introduction
    variety range amount majority number source one two three four five six seven eight nine ten
    several many few both half whole pair couple dozen""".split()
)

# Modifiers that are stripped from the *left* edge of a candidate phrase.
LEADING_MODIFIERS = frozenset(
    """such each every this that these those other another same several various many most
    more some any all both own certain different additional following first second third
    next previous simple basic common main key important given particular specific new
    typical general possible single better best good high low similar corresponding
    relevant available potential entire whole large small major minor useful popular
    efficient effective""".split()
)

# Frequent verbs/adverbs. They act as phrase breakers when no parser is available
# and are excluded from topic-model vocabularies.
VERBISH = frozenset(
    """use uses used using make makes made take takes taken give gives given show shows
    shown find finds found get gets need needs needed require requires required allow
    allows allowed provide provides provided produce produces produced contain contains
    contained include includes included consist consists depend depends compute computes
    computed calculate calculates calculated represent represents represented define
    defines defined call calls called refer refers denote denotes mean means become
    becomes apply applies applied assign assigns assigned estimate estimates estimated
    learn learns learned predict predicts predicted measure measures measured evaluate
    evaluates evaluated improve improves improved reduce reduces reduced help helps helped
    work works worked try tries tried see sees seen let lets say says said split splits
    identify identifies identified extract extracts extracted convert converts converted
    map maps mapped generate generates generated build builds built create creates
    created perform performs performed handle handles handled solve solves solved avoid
    avoids avoided capture captures captured remove removes removed assume assumes
    assumed combine combines combined choose chooses chosen select selects selected
    increase increases increased decrease decreases decreased obtain obtains obtained
    occur occurs occurred appear appears appeared follow follows followed form forms
    formed involve involves involved lead leads led keep keeps kept put puts run runs
    ran start starts started end ends ended base bases based rely relies relied tend
    tends tended look looks looked know knows known think thinks thought consider
    considers considered note notes noted describe describes described explain explains
    explained discuss discusses discussed introduce introduces introduced address
    addresses addressed ensure ensures ensured enable enables enabled support supports
    supported expect expects expected want wants wanted hold holds held change changes
    changed test tests tested compare compares compared count counts counted
    also often usually typically generally simply only just then thus hence therefore
    first next finally instead rather even still already never together raw unseen express
    treat treats treated condition conditions conditioned deletes delete adds add move moves
    falls fall store stores stored assigns lowers lower higher mean pay input""".split()
)

SMALL_WORDS = frozenset({"of", "and", "the", "a", "an", "in", "for", "to", "on", "by", "with", "or", "vs"})

# Nouns after "is a ___ of" that signal taxonomy vs. part-whole relations.
TYPE_NOUNS = frozenset(
    {"type", "kind", "form", "variant", "instance", "example", "class", "subclass", "subtype",
     "case", "subset", "version", "family"}
)
PART_NOUNS = frozenset({"part", "component", "member", "element", "step", "stage", "ingredient"})

# ---------------------------------------------------------------------------
# Relation lexicon.  Each rule: (regex, relation type, prerequisite direction)
# prerequisite direction:
#   "obj_to_subj": the OBJECT must be learned before the SUBJECT
#   "subj_to_obj": the SUBJECT must be learned before the OBJECT
#   None         : no ordering implied
# ---------------------------------------------------------------------------
_R = re.compile
RELATION_RULES = [
    (_R(r"\b(is|are|be|was|were|been)\s+(used|employed|applied|utili[sz]ed)\b"), "used_for", "subj_to_obj"),
    (_R(r"\b(part|component|member|subset|element|step|stage|ingredient)\s+of\b|\bbelongs?\s+to\b"),
     "part_of", "subj_to_obj"),
    (_R(r"\b(consist\w*|compos\w*|compris\w*|contain\w*|includ\w*)\b"), "has_part", "obj_to_subj"),
    (_R(r"\b(type|kind|form|variant|instance|example|class|subclass|subtype|case|version|family)\s+of\b"
        r"|^(is|are|be|was|were)(\s+an?)?$|\b(is|are|be)\s+an?\b"), "is_a", "obj_to_subj"),
    (_R(r"\b(called|known\s+as|defined\s+as|refers?\s+to|denotes?|stands?\s+for)\b"), "defined_as", None),
    (_R(r"\b(uses?|used|using|requires?|required|depend\w*|rel(?:y|ies|ied|ying)|needs?|needed|builds?|built|"
        r"bas(?:e|es|ed|ing)|appl(?:y|ies|ied|ying)|employs?|employed|utili[sz]\w*|leverag\w*|assum\w*|"
        r"operat\w*|deriv\w*|inherit\w*|extend\w*|exploit\w*|incorporat\w*|via|through)\b"),
     "requires", "obj_to_subj"),
    (_R(r"\b(evaluat\w*|measur\w*|assess\w*|validat\w*|scor\w*|test\w*|compar\w*|quantif\w*|judg\w*)\b"),
     "evaluates", None),
    (_R(r"\b(improv\w*|handl\w*|solv\w*|address\w*|prevent\w*|avoid\w*|mitigat\w*|smooth\w*|"
        r"regulari[sz]\w*|overcom\w*|fix\w*|correct\w*|reduc\w*)\b"), "improves", None),
    (_R(r"\b(produc\w*|generat\w*|yield\w*|output\w*|creat\w*|return\w*|result\w*|give\w*|provid\w*|"
        r"offer\w*|enabl\w*|allow\w*|support\w*)\b"), "produces", None),
    (_R(r"\b(extract\w*|identif\w*|detect\w*|discover\w*|find\w*|found|learn\w*|predict\w*|estimat\w*|"
        r"comput\w*|calculat\w*|represent\w*|encod\w*|map\w*|assign\w*|split\w*|segment\w*|convert\w*|"
        r"transform\w*|normali[sz]\w*|break\w*|divid\w*|tag\w*|pars\w*|analy[sz]\w*|process\w*|count\w*|"
        r"remov\w*|filter\w*)\b"), "operates_on", None),
]


def classify_relation(rel: str):
    """Return (relation_type, prerequisite_direction) for a relation phrase."""
    s = rel.lower().strip()
    for rx, rtype, pdir in RELATION_RULES:
        if rx.search(s):
            return rtype, pdir
    return "related", None


REL_LABELS = {
    "requires": "requires / uses",
    "is_a": "is a (taxonomy)",
    "has_part": "has part",
    "part_of": "part of",
    "used_for": "used for",
    "defined_as": "defined as",
    "evaluates": "evaluates / measures",
    "improves": "improves / fixes",
    "produces": "produces / enables",
    "operates_on": "operates on",
    "related": "related",
    "subtopic": "has sub-topic",
    "covers": "covers (same section)",
}

REL_COLORS = {
    "subtopic": "#22d3ee",
    "covers": "#a78bfa",
    "requires": "#f59e0b",
    "is_a": "#38bdf8",
    "has_part": "#a78bfa",
    "part_of": "#a78bfa",
    "used_for": "#34d399",
    "defined_as": "#94a3b8",
    "evaluates": "#f472b6",
    "improves": "#fb7185",
    "produces": "#2dd4bf",
    "operates_on": "#facc15",
    "related": "#94a3b8",
}

TOPIC_PALETTE = [
    "#4F9DFF", "#FF8A4C", "#3DD6A5", "#E56BFF", "#FFD166",
    "#FF6B81", "#7ED957", "#8E9BFF", "#26C6DA", "#FFB74D",
    "#C0CA33", "#9575CD",
]
UNASSIGNED_COLOR = "#94a3b8"
PATH_COLOR = "#ff3b3b"


# Extra everyday verbs (and number words) so that, without a parser, they never become concepts.
_MORE_VERBS = """carry happen raise rise slow grow become begin bring cause decide enter fail feel fill hit join
keep know leave live lose offer open pay pull push reach receive remain seek sell send sit speak spend stay
succeed suggest tell throw understand wait want watch win lead protect prevent encourage limit check pump
produce replace power connect appoint follow change ask answer happen happens""".split()
for _v in _MORE_VERBS:
    for _f in (_v, _v + "s", _v + "es", _v[:-1] + "ies" if _v.endswith("y") else _v + "s"):
        VERBISH = VERBISH | {_f}
LEADING_MODIFIERS = LEADING_MODIFIERS | frozenset(
    "one two three four five six seven eight nine ten hundred thousand".split())
GENERIC_SINGLE = GENERIC_SINGLE | frozenset("hour day week month year minute second century period".split())
