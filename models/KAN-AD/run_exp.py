from EasyTSAD.Controller import TSADController
from kanad import KANAD  # noqa

if __name__ == "__main__":
    # Create a global controller
    gctrl = TSADController()

    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    dataset_types = "UTS"

    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="datasets",
        datasets=datasets,
    )

    training_schema = "naive"
    method = "KANAD"  # string of your algo class

    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="kanad/config.toml",
    )

    """============= [EVALUATION SETTINGS] ============="""

    from EasyTSAD.Evaluations.Protocols import (
        EventF1PA,
        PointF1PA,
        PointKthF1PA,
        PointAuprcPA,
    )

    # Specifying evaluation protocols
    gctrl.set_evals(
        [PointF1PA(), EventF1PA(mode="squeeze"), PointKthF1PA(k=5), PointAuprcPA()]
    )

    gctrl.do_evals(method=method, training_schema=training_schema)
