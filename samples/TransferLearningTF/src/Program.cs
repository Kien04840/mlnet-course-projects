using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TransferLearningTF
{
    public class Program
    {
        // ---------- ASSET PATHS ----------
        static string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static string _imagesFolder = Path.Combine(_assetsPath, "images");
        static string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

        // ---------- DATA CLASSES ----------
        public class ImageData
        {
            [LoadColumn(0)]
            public string? ImagePath; // expected to be file name (e.g. "toaster3.jpg") as in tags.tsv

            [LoadColumn(1)]
            public string? Label;
        }

        public class ImagePrediction : ImageData
        {
            // TF penultimate features (tên cột giữ nguyên vì CreateEnumerable dùng nó cho hiển thị test)
            [ColumnName("softmax2_pre_activation")]
            public float[]? Features;

            // THIS is the probabilities output by the ML.NET trainer (values in 0..1)
            [ColumnName("Score")]
            public float[]? Score;

            public string? PredictedLabelValue;
        }


        // ---------- INCEPTION SETTINGS ----------
        struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }



        static void Main(string[] args)
        {
            Console.WriteLine("=============== Training classification model ===============");

            MLContext mlContext = new MLContext(seed: 1);

            // Generate/train model
            ITransformer model = GenerateModel(mlContext);

            Console.WriteLine("=============== Making single image classification ===============");

            // Single image classification & display
            ClassifySingleImage(mlContext, model);

            Console.WriteLine("Done.");
        }

        // ------------------ GenerateModel: build pipeline, train, evaluate ------------------
        static ITransformer GenerateModel(MLContext mlContext)
        {
            // 1) Define pipeline: load images, resize, extract pixels
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(
                        outputColumnName: "input",
                        imageFolder: _imagesFolder,
                        inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(
                        outputColumnName: "input",
                        imageWidth: InceptionSettings.ImageWidth,
                        imageHeight: InceptionSettings.ImageHeight,
                        inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                        outputColumnName: "input",
                        interleavePixelColors: InceptionSettings.ChannelsLast,
                        offsetImage: InceptionSettings.Mean));

            // 2) Load TensorFlow model and score (remove final layer — use penultimate features)
            pipeline = pipeline.Append(
                mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2_pre_activation" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true));

            // 3) Convert string labels to keys for trainer
            pipeline = pipeline.Append(mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "LabelKey", inputColumnName: "Label"));

            // 4) Add trainer (use multiclass trainer on TF features)
            pipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers
                .LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation")
            );

            // 5) Convert predicted keys back to original labels
            pipeline = pipeline.Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabelValue", inputColumnName: "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            // ------------------ Train ------------------
            // Load training data (tab-separated: file \t label)
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(
                path: _trainTagsTsv, hasHeader: false, separatorChar: '\t');

            ITransformer model = pipeline.Fit(trainingData);

            // ------------------ Evaluate ------------------
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(
                path: _testTagsTsv, hasHeader: false, separatorChar: '\t');

            IDataView predictions = model.Transform(testData);

            // Create IEnumerable for display
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: true);
            DisplayResults(imagePredictionData);

            // Evaluate metrics (requires label column in key format: "LabelKey")
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine("=============== Classification metrics ===============");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        // ------------------ Classify single image ------------------
        static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = Path.GetFileName(_predictSingleImage) // tags.tsv expects filename; prediction engine expects same schema
            };

            // Create prediction engine (single-threaded convenience)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            // Display the prediction as requested
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
        }

        // ------------------ Display utility ------------------
        static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
            }
        }
    }
}
