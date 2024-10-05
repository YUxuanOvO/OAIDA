import pandas as pd
from causal_discovery.ldecc1 import LDECCAlgorithm


def select(data, treatment_node, alpha=0.05, use_ci_oracle=False,
           graph_true=None, enable_logging=True, ldecc_do_checks=False, max_tests=20000):
    """
    Perform feature selection using LDECC algorithm.

    Parameters:
        data (pd.DataFrame): Input dataset.
        treatment_node (str): Name of the treatment node (class label).
        outcome_node (str): Name of the outcome node (target feature).
        alpha (float): Significance level for conditional independence tests.
        use_ci_oracle (bool): Whether to use oracle conditional independence tests.
        graph_true: Ground truth graph (optional).
        enable_logging (bool): Enable logging for the LDECC algorithm.
        ldecc_do_checks (bool): Enable checks during LDECC algorithm execution.
        max_tests: Maximum number of conditional independence tests to perform (optional).

    Returns:
        selected_features (list): List of selected features (parent and child nodes).
    """

    # Instantiate LDECCAlgorithm
    causal_discovery = LDECCAlgorithm(
        treatment_node, alpha, use_ci_oracle,
        graph_true, enable_logging, ldecc_do_checks, max_tests)

    # Run causal discovery algorithm
    result = causal_discovery.run(data)

    # Extract selected features (parent and child nodes)
    tmt_parents = result["tmt_parents"]
    tmt_children = result["tmt_children"]
    to_be_oriented = result["unoriented"]
    # Combine parent and child nodes to get selected features
    features = list(set(tmt_parents).union(tmt_children).union(to_be_oriented))

    return features

# Example usage:
# selected_features = select(data, 'item', 'Label', alpha=0.05, use_ci_oracle=False)
