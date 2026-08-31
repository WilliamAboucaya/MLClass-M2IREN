"""
Synthetic student grades dataset generator.

Designed for a course progressing from linear regression -> polynomial /
interaction regression -> random forest, with each method expected to fit
progressively better because of the effects baked into the data:

Features
--------
- hours_studied     : float in [0, 10], uniformly distributed.
- sleep_hours       : float in [0, 10], concentrated between 6 and 8
                       (truncated normal, mean=7, std=1.3).
- class_attendance  : int in [0, 6], binomial(n=6, p=0.7) so most students
                       attend fairly regularly but there's still spread.

Effects on grade (target, clipped to [0, 20])
-----------------------------------------------
1. class_attendance   : purely LINEAR positive effect.
2. hours_studied      : concave QUADRATIC effect (diminishing returns -
                        studying more always helps, but less and less).
3. sleep_hours        : inverted-U QUADRATIC effect (an effect around 7h
                        sleep is optimal; too little or too much hurts).
4. INTERACTION (2-way): the payoff from studying is conditional on sleep.
                        A piecewise ("kinked") multiplier scales down the
                        studying effect when sleep_hours is below ~7h,
                        and saturates at full effectiveness above that.
5. INTERACTION (3-way): an "engaged student" bonus that only kicks in
                        when class_attendance is high AND hours_studied
                        is high AND sleep_hours is in a healthy range,
                        all at once. This is a genuine conjunctive/
                        threshold (AND-type) effect - structurally
                        impossible for a polynomial regression to
                        represent exactly, but trivial for a tree-based
                        model to pick up by splitting on each condition.

A pedagogical progression is expected on held-out data:
    Linear regression         -> decent but biased (misses curvature/kinks)
    Polynomial + interactions -> better (captures quadratics + 2-way x)
    Random forest             -> best (also captures the 3-way AND rule)
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

RANDOM_SEED = 1234


def generate_student_grades(n_samples: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic student grades dataset.

    Parameters
    ----------
    n_samples : int
        Number of student records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: hours_studied, sleep_hours, class_attendance, grade
    """
    rng = np.random.default_rng(seed)

    hours_studied = rng.uniform(0, 10, n_samples)

    sleep_hours = rng.normal(loc=7.0, scale=1.3, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 0, 10)

    class_attendance = rng.binomial(n=6, p=0.7, size=n_samples)

    base = 2.5

    # 1) Linear effect of attendance (each class attended is worth ~1.1 pts)
    attendance_effect = 1.1 * class_attendance

    # 2) Conditional 2-way interaction: studying pays off more when
    #    well-rested. Effectiveness multiplier ramps linearly from 0.3
    #    (very little sleep) up to 1.0 once sleep_hours reaches 7h, then
    #    plateaus.
    study_effectiveness = 0.3 + 0.7 * np.clip(sleep_hours / 7.0, 0, 1)

    # 3) Concave (diminishing-returns) quadratic effect of hours studied,
    #    scaled by how effective that studying actually is.
    raw_study_effect = 1.4 * hours_studied - 0.07 * hours_studied ** 2
    study_effect = raw_study_effect * study_effectiveness

    # 4) Inverted-U quadratic effect of sleep around an optimum of 7h.
    sleep_effect = -0.2 * (sleep_hours - 7.0) ** 2

    # 5) "Engaged student" 3-way conjunctive bonus: only awarded when
    #    attendance, study time AND sleep are simultaneously in a good
    #    range. A pure AND/step rule - impossible for a polynomial
    #    regression to represent exactly, easy for a random forest.
    engaged = (
        (class_attendance >= 5)
        & (hours_studied >= 6)
        & (sleep_hours >= 5.5) & (sleep_hours <= 8.5)
    )
    engaged_bonus = np.where(engaged, 3.0, 0.0)

    noise = rng.normal(loc=0.0, scale=0.9, size=n_samples)

    grade = (
        base + attendance_effect + study_effect + sleep_effect
        + engaged_bonus + noise
    )
    grade = np.clip(grade, 0, 20)
    grade = (grade * 2).round() / 2

    df = pd.DataFrame({
        "hours_studied": np.round(hours_studied, 2),
        "sleep_hours": np.round(sleep_hours, 2),
        "class_attendance": class_attendance.astype(int),
        "grade": np.round(grade, 2),
    })

    return df


if __name__ == "__main__":
    df = generate_student_grades(n_samples=1500)
    df.to_csv("synthetic_student_data.csv", index=False, encoding="utf-8")
    print(df.describe())
    print("\nSaved to student_grades.csv")

    X = df[["hours_studied", "sleep_hours", "class_attendance"]]
    y = df["grade"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    models = {
        "Linear regression": LinearRegression(),
        "Polynomial (deg=2) + interactions": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_SEED
        ),
    }

    print("\nModel comparison (R^2 on held-out test set):")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"  {name:<35s}: R^2 = {r2_score(y_test, preds):.3f}")