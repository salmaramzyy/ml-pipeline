import mlflow
import sys

mlflow.set_tracking_uri("file:./mlruns")

THRESHOLD = 0.85

# Read run_id
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print("Accuracy:", accuracy)

if accuracy is None:
    print("No accuracy found")
    sys.exit(1)

if accuracy < THRESHOLD:
    print("Model failed threshold")
    sys.exit(1)
else:
    print("Model passed")