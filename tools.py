"""
tools.py — OpenAI-compatible tool definitions for the Video Explorer agents.
"""

from config import MIN_EXPAND_SPAN, GRID_K

# ==========================================
# TOOL DEFINITIONS
# ==========================================
TOOL_EXPAND = {
    "type": "function",
    "function": {
        "name": "EXPAND",
        "description": "Expand into a specific cell for a new grid covering a closer temporal view.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "integer"}
            },
            "required": ["cell_id"]
        }
    }
}

TOOL_BACKTRACK = {
    "type": "function",
    "function": {
        "name": "BACKTRACK",
        "description": "Go back to the parent view within your assigned region.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}

TOOL_ZOOM = {
    "type": "function",
    "function": {
        "name": "ZOOM",
        "description": (
            "Use only when you found a relevant scene or anchor and need a closer, "
            "high-resolution look at the details. Do NOT use for general exploration — "
            "only when you spotted something promising and need to confirm or read fine details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "duration":  {"type": "number"}
            },
            "required": ["timestamp"]
        }
    }
}

TOOL_ADD_TO_SCRATCHPAD = {
    "type": "function",
    "function": {
        "name": "ADD_TO_SCRATCHPAD",
        "description": "Save evidence that helps complete the SEARCH TASK. Only save if it directly helps answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp":   {"type": "number"},
                            "description": {
                                "type": "string",
                                "description": "Format: '<what you observe>. This helps the search task because <how it helps find the answer>.'"
                            },
                            "confidence": {
                                "type": "number",
                                "description": (
                                    "0.5=tangential context, 0.6=mentions topic, "
                                    "0.7=relevant but indirect, 0.8=directly addresses search task, "
                                    "0.9=strong direct evidence, 1.0=definitive answer visible"
                                )
                            }
                        },
                        "required": ["timestamp", "description", "confidence"]
                    }
                }
            },
            "required": ["items"]
        }
    }
}

TOOL_INVESTIGATE = {
    "type": "function",
    "function": {
        "name": "INVESTIGATE",
        "description": (
            "Use only when you found the exact anchor scene the search task describes, "
            "but the answer is in the neighboring frames. Jumps to show what happens "
            "RIGHT BEFORE or AFTER a timestamp. Do NOT use for general exploration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "number",
                    "description": "The anchor timestamp you found."
                },
                "direction": {
                    "type": "string",
                    "enum": ["before", "after"],
                    "description": "'after' = what happens next, 'before' = what led to this."
                },
                "reason": {
                    "type": "string",
                    "description": "Why you need to look before/after."
                }
            },
            "required": ["timestamp", "direction", "reason"]
        }
    }
}

TOOL_FINISHED = {
    "type": "function",
    "function": {
        "name": "FINISHED",
        "description": (
            "Declare this region fully explored. Call when no more relevant evidence "
            "can be found, or when you have found sufficient evidence to answer the query."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}

TOOL_MARK_PROMISING = {
    "type": "function",
    "function": {
        "name": "MARK_PROMISING",
        "description": (
            "Mark cell/s in the current grid as promising for further exploration later. "
            "These cells will be added to the exploration queue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cell_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 2
                }
            },
            "required": ["cell_ids"]
        }
    }
}


# ==========================================
# TOOL SET BUILDERS
# ==========================================
def get_master_tools():
    """Master can only: ADD_TO_SCRATCHPAD, FINISHED."""
    return [TOOL_ADD_TO_SCRATCHPAD, TOOL_FINISHED]


def get_exploration_tools(span, depth, max_depth=None):
    """
    Full worker tool set.
    - EXPAND: only if span >= MIN_EXPAND_SPAN AND depth < max_depth
    - BACKTRACK: only if depth > 0
    - All others: always available
    """
    tools      = []
    can_expand = span >= MIN_EXPAND_SPAN
    if max_depth is not None and depth >= max_depth:
        can_expand = False

    if can_expand:
        tools.append(TOOL_EXPAND)
    if depth > 0:
        tools.append(TOOL_BACKTRACK)

    tools.extend([
        TOOL_ZOOM,
        TOOL_INVESTIGATE,
        TOOL_ADD_TO_SCRATCHPAD,
        TOOL_MARK_PROMISING,
        TOOL_FINISHED
    ])
    return tools
