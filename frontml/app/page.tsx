"use client";

import { useState, FormEvent } from "react";
import axios from "axios";
import styles from "./page.module.css";

type ResultState = {
  error?: string;
  best_params?: any;
  mean_squared_error?: number;
  y_test?: number[];
  y_pred?: number[];
  plot_image?: string;
} | null;

const modelOptions = {
  classification: [
    "Decision Tree",
    "Elastic Net",
    "Random Forest",
    "SVM",
    "Bagging",
  ],
  regression: [
    "Decision Tree Regressor",
    "Elastic Net Regressor",
    "SVR",
    "Random Forest Regressor",
    "Bagging Regressor",
  ],
};

export default function Home() {
  const [modelType, setModelType] = useState("null");
  const [modelName, setModelName] = useState("null");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [results, setResults] = useState<ResultState>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files ? event.target.files[0] : null;
    setCsvFile(file);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!csvFile || modelType === "null" || modelName === "null") {
      console.error("Form is incomplete.");
      return;
    }

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("model", modelName);

    try {
      const endpoint = `http://localhost:5000/${
        modelType === "classification"
          ? "evaluate_classification"
          : "evaluate_regressor"
      }`;
      const response = await axios.post(endpoint, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResults(response.data);
    } catch (error) {
      console.error("There was an error with the request:", error);
      setResults({
        error:
          "Failed to fetch results. Please check the console for more information.",
      });
    }
  };

  return (
    <main className={styles.main}>
      <h1>Evaluate Machine Learning Model</h1>
      <form onSubmit={handleSubmit} className={styles.form}>
        <div>
          <label>
            Model Type:
            <select
              value={modelType}
              onChange={(e) => {
                setModelType(e.target.value);
                setModelName("null");
              }}
            >
              <option value="null">Select an Option</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
        </div>
        <div>
          <label>
            Model Name:
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={modelType === "null"}
            >
              <option value="null">Select a Model</option>
              {modelType !== "null" &&
                modelOptions[modelType as keyof typeof modelOptions].map(
                  (model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  )
                )}
            </select>
          </label>
        </div>
        {modelType !== "null" && (
          <div>
            <label>
              Upload CSV:
              <input type="file" onChange={handleFileChange} />
            </label>
          </div>
        )}

        <button type="submit">Submit</button>
      </form>
      {results && results.error && <p>Error: {results.error}</p>}
      {results && results.mean_squared_error && (
        <p>Mean Squared Error: {results.mean_squared_error}</p>
      )}
      {results && results.plot_image && (
        <div className={styles.results}>
          <img
            src={`data:image/png;base64,${results.plot_image}`}
            alt="Plot Image"
          />
        </div>
      )}
    </main>
  );
}
