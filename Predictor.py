"""
Student Score Predictor - Final AI & ML Project

Predicts a student's final exam score from study habits using
Machine Learning (Scikit-learn). Compares two models, shows evaluation
metrics, saves a chart, and lets the user predict their own score.

Note: the dataset is synthetic (generated in code) so the project runs
anywhere without downloading files.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["hours_studied", "attendance", "previous_score", "sleep_hours"]


def create_dataset(n=300, seed=42):
    """Generate a realistic synthetic student dataset."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(1, 10, n)
    attendance = rng.uniform(50, 100, n)
    previous = rng.uniform(35, 95, n)
    sleep = rng.uniform(4, 9, n)
    noise = rng.normal(0, 4, n)
    final = (
        0.45 * previous + 3.2 * hours + 0.18 * attendance + 1.2 * sleep + noise - 5
    )
    final = np.clip(final, 0, 100)
    return pd.DataFrame({
        "hours_studied": hours.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous.round(1),
        "sleep_hours": sleep.round(1),
        "final_score": final.round(1),
    })


def train_and_compare(df):
    """Train two models and return the best one with a results table."""
    X = df[FEATURES]
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results, trained = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE": round(mean_absolute_error(y_test, pred), 2),
            "R2": round(r2_score(y_test, pred), 3),
        })
        trained[name] = (model, pred)
    results_df = pd.DataFrame(results)
    best_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
    return trained[best_name][0], best_name, results_df, y_test, trained[best_name][1]


def save_chart(y_test, y_pred, model_name):
    """Save an actual vs predicted scatter plot."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color="#4C72B0")
    plt.plot([0, 100], [0, 100], "r--", label="Perfect prediction")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    plt.close()


def read_number(prompt, low, high):
    """Ask until the user enters a number between low and high."""
    while True:
        try:
            value = float(input(prompt))
            if low <= value <= high:
                return value
            print(f"  Please enter a value between {low} and {high}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def main():
    print("=== Student Score Predictor ===")
    df = create_dataset()
    df.to_csv("student_data.csv", index=False)
    print(f"Dataset created: {df.shape[0]} students, saved to student_data.csv")

    model, best_name, results, y_test, y_pred = train_and_compare(df)
    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    save_chart(y_test, y_pred, best_name)
    print("Chart saved: actual_vs_predicted.png\n")

    while True:
        print("Enter student details:")
        hours = read_number("  Hours studied per day (0-12): ", 0, 12)
        attendance = read_number("  Attendance % (0-100): ", 0, 100)
        previous = read_number("  Previous exam score (0-100): ", 0, 100)
        sleep = read_number("  Sleep hours per day (0-12): ", 0, 12)

        sample = pd.DataFrame([[hours, attendance, previous, sleep]], columns=FEATURES)
        score = float(np.clip(model.predict(sample)[0], 0, 100))
        print(f"\nPredicted final score: {score:.1f} / 100\n")

        if input("Predict another student? (y/n): ").strip().lower() != "y":
            print("Thank you for using the predictor!")
            break
        print()


if __name__ == "__main__":
    main()
