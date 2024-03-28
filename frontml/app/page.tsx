"use client";

import { useState, FormEvent } from "react";
import axios from "axios";
import styles from "./page.module.css";
import {
  FormControl,
  Input,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from "@mui/material";

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
      <div className={styles.border}>
        <h1 className={styles.title}>Evaluate Machine Learning Model</h1>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <FormControl variant="outlined" fullWidth>
              <InputLabel id="model-type-label">Model Type</InputLabel>
              <Select
                labelId="model-type-label"
                id="model-type"
                value={modelType}
                onChange={(e) => {
                  setModelType(e.target.value as string);
                  setModelName("null");
                }}
                label="Model Type"
              >
                <MenuItem value="null">Select an Option</MenuItem>
                <MenuItem value="classification">Classification</MenuItem>
                <MenuItem value="regression">Regression</MenuItem>
              </Select>
            </FormControl>
          </div>
          <div className={styles.field}>
            <FormControl
              variant="outlined"
              fullWidth
              disabled={modelType === "null"}
            >
              <InputLabel id="model-name-label">Model Name</InputLabel>
              <Select
                labelId="model-name-label"
                id="model-name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value as string)}
                label="Model Name"
              >
                <MenuItem value="null">Select a Model</MenuItem>
                {modelType !== "null" &&
                  modelOptions[modelType as keyof typeof modelOptions].map(
                    (model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    )
                  )}
              </Select>
            </FormControl>
          </div>
          {modelType !== "null" && (
            <div className={styles.field}>
              <Input
                id="file-upload"
                type="file"
                onChange={handleFileChange}
                disableUnderline
                style={{ display: "none" }}
              />
              <label htmlFor="file-upload" className={styles.uploadButton}>
                Upload File
              </label>
            </div>
          )}

          <Button
            variant="contained"
            color="primary"
            type="submit"
            className={styles.customButton}
          >
            Submit
          </Button>
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
      </div>
    </main>
  );
}
