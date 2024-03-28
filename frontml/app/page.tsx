'use client';
import { useState, useEffect, FormEvent } from 'react';
import axios from 'axios';
import Chart from 'chart.js/auto';
import styles from './page.module.css';
import { printTreeView } from 'next/dist/build/utils';

type ResultState = {
  error?: string;
  best_params?: any;
  mean_squared_error?: number;
  y_test?: number[];
  y_pred?: number[];
} | null;

export default function Home() {
  const [modelType, setModelType] = useState('classification');
  const [modelName, setModelName] = useState('');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [results, setResults] = useState<ResultState>(null);
  const [fileData, setFileData] = useState<number[]>([]);

  useEffect(() => {
    if (csvFile) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target) {
          const csvData = event.target.result as string;
          const lines = csvData.split('\n');
          const data: number[] = [];
          for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length >= 31) {
              let kk;
              if (values[30] === '"0"')
                kk = 1;
              else
                kk = 0;
              data.push(kk); 
            }
          }
          setFileData(data);
        }
      };
      reader.readAsText(csvFile);
    }
  }, [csvFile]);
  

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files ? event.target.files[0] : null;
    setCsvFile(file);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!csvFile) {
      console.error("No CSV file selected.");
      return;
    }

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("model", modelName);

    try {
      const endpoint = `http://localhost:5000/${modelType === 'classification' ? 'evaluate_classification' : 'evaluate_regressor'}`;
      const response = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response);
      setResults(response.data);
    } catch (error) {
      console.error("There was an error with the request:", error);
      setResults({ error: "Failed to fetch results. Please check the console for more information." });
    }
  };

  useEffect(() => {
    if (results && results.y_pred && fileData.length > 0) {
      console.log(results.y_pred);
      const ctx = document.getElementById('myChart') as HTMLCanvasElement;
      new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Actual vs Predicted',
              data: fileData.map((value, index) => ({ x: value, y: results.y_pred![index] })),
              borderColor: 'blue',
              backgroundColor: 'blue',
            },
            {
              label: 'Ideal',
              data: fileData.map((value) => ({ x: value, y: value })),
              borderColor: 'red',
              backgroundColor: 'red',
            },
          ],
        },
        options: {
          scales: {
            x: {
              type: 'linear',
              position: 'bottom',
              title: {
                display: true,
                text: 'Actual',
              },
            },
            y: {
              type: 'linear',
              position: 'left',
              title: {
                display: true,
                text: 'Predicted',
              },
            },
          },
        },
      });
    }
  }, [results, fileData]);

  return (
    <main className={styles.main}>
      <h1>Evaluate Machine Learning Model</h1>
      <form onSubmit={handleSubmit} className={styles.form}>
        <div>
          <label>
            Model Type:
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
        </div>
        <div>
          <label>
            Model Name:
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="Model Name" />
          </label>
        </div>
        <div>
          <label>
            Upload CSV:
            <input type="file" onChange={handleFileChange} />
          </label>
        </div>
        <button type="submit">Submit</button>
      </form>
      <div className={styles.results}>
        <canvas id="myChart" />
      </div>
    </main>
  );
}
