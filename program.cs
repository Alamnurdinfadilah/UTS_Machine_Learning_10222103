using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HoaxDetector
{
    public class ModelInput
    {
        // CSV columns: ID,label,tanggal,judul,narasi,nama file gambar
        [LoadColumn(1)] public float Label { get; set; }
        [LoadColumn(3)] public string Judul { get; set; }
        [LoadColumn(4)] public string Narasi { get; set; }
    }

    public class ModelOutputBinary
    {
        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    public class ModelOutputMulticlass
    {
        [ColumnName("PredictedLabel")] public string PredictedLabel { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var dataPath = args.Length > 0 ? args[0] : "Data_latih.csv";
            var modelPath = args.Length > 1 ? args[1] : "hoax_model.zip";

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Data file not found: {dataPath}");
                return;
            }

            var mlContext = new MLContext(seed: 0);

            // Load data
            var data = mlContext.Data.LoadFromTextFile<ModelInput>(path: dataPath, hasHeader: true, separatorChar: ',');

            // Quick label analysis to decide binary/multiclass
            var labels = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: false)
                .Select(r => r.Label).Distinct().OrderBy(x => x).ToArray();

            Console.WriteLine($"Detected label values: {string.Join(", ", labels)}");
            bool isBinary = labels.Length == 2;

            // Create Text column by concatenating Judul + Narasi
            var pipelinePrep = mlContext.Transforms.CustomMapping((Action<ModelInput, TextOutput>)((input, output) =>
            {
                output.Text = string.Concat(input?.Judul ?? string.Empty, "\n", input?.Narasi ?? string.Empty);
            }), contractName: null)
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text"));

            if (isBinary)
            {
                Console.WriteLine("Using binary classification pipeline (SdcaLogisticRegression).");
                var pipeline = pipelinePrep
                    .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Label"))
                    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

                var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
                var trainData = split.TrainSet;
                var testData = split.TestSet;

                Console.WriteLine("Training...");
                var model = pipeline.Fit(trainData);

                Console.WriteLine("Evaluating...");
                var predictions = model.Transform(testData);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
                Console.WriteLine($"Precision: {metrics.PositivePrecision:P2}");
                Console.WriteLine($"Recall: {metrics.PositiveRecall:P2}");

                // Save model
                mlContext.Model.Save(model, trainData.Schema, modelPath);
                Console.WriteLine($"Model saved to {modelPath}");

                // Example prediction
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutputBinary>(model);
                var sample = new ModelInput { Judul = "Contoh judul", Narasi = "Isi berita yang mengandung klaim tidak berdasar." };
                var result = predEngine.Predict(sample);
                Console.WriteLine($"Sample prediction: {(result.PredictedLabel ? "Hoax" : "Bukan Hoax")} (Prob: {result.Probability:P2})");
            }
            else
            {
                Console.WriteLine("Using multiclass classification pipeline (SdcaMaximumEntropy).");
                // For multiclass, map label to key
                var pipeline = pipelinePrep
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

                var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
                var trainData = split.TrainSet;
                var testData = split.TestSet;

                Console.WriteLine("Training...");
                var model = pipeline.Fit(trainData);

                Console.WriteLine("Evaluating...");
                var predictions = model.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey");

                Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
                Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
                Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");

                // Save model
                mlContext.Model.Save(model, trainData.Schema, modelPath);
                Console.WriteLine($"Model saved to {modelPath}");

                // Example prediction
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutputMulticlass>(model);
                var sample = new ModelInput { Judul = "Contoh judul", Narasi = "Isi berita yang mengandung klaim tidak berdasar." };
                var result = predEngine.Predict(sample);
                Console.WriteLine($"Sample prediction: {result.PredictedLabel}");
            }
        }

        private class TextOutput { public string Text { get; set; } }
    }
}
