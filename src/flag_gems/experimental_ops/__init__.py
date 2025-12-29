from flag_gems.experimental_ops.rms_norm import rms_norm


def get_operator_mappings(experimental_ops=None):
    """Return operator mappings with experimental_ops replacements when enabled."""
    experimental_ops = []
    if experimental_ops is not None:
        if experimental_ops == "all":
            experimental_ops = ["rms_norm"]
        elif isinstance(experimental_ops, list):
            experimental_ops = experimental_ops

    # Return mappings for experimental_ops operators only
    experimental_ops_mappings = []
    if "rms_norm" in experimental_ops:
        experimental_ops_mappings.append(("rms_norm", rms_norm))

    return experimental_ops_mappings


__all__ = ["get_operator_mappings", "rms_norm"]
