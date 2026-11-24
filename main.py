import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Fraud Detection System on port {port}...")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()