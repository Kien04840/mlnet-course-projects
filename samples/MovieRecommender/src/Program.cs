using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
// Đảm bảo namespace này khớp với tên dự án của bạn
using MovieRecommender;

namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            // Khởi tạo MLContext
            MLContext mlContext = new MLContext();

            // 1. Tải Dữ liệu
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            // 2. Xây dựng và Huấn luyện Mô hình
            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            // 3. Đánh giá Mô hình
            EvaluateModel(mlContext, testDataView, model);

            // 4. Dự đoán đơn lẻ
            UseModelForSinglePrediction(mlContext, model);

            // 5. Lưu Mô hình
            SaveModel(mlContext, trainingDataView.Schema, model);
        }

        // === Phương thức 1: Tải Dữ liệu ===
        static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            // Tải dữ liệu từ file CSV
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        // === Phương thức 2: Xây dựng và Huấn luyện Mô hình ===
        static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            // Định nghĩa pipeline biến đổi dữ liệu:
            // - MapValueToKey: Chuyển đổi userId và movieId sang dạng numeric key (cần thiết cho Matrix Factorization)
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            // Cài đặt thuật toán Matrix Factorization
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20, // Số lần lặp
                ApproximationRank = 100 // Kích thước của ma trận ẩn
            };

            // Thêm trainer vào cuối pipeline
            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            // Huấn luyện mô hình
            ITransformer model = trainerEstimator.Fit(trainingDataView);
            Console.WriteLine("=============== End of training ==================");

            return model;
        }

        // === Phương thức 3: Đánh giá Mô hình ===
        static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");

            // Biến đổi dữ liệu test bằng mô hình đã train để dự đoán
            var prediction = model.Transform(testDataView);

            // Đánh giá mô hình (sử dụng Regression Metrics)
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError:0.00}");
            Console.WriteLine($"RSquared: {metrics.RSquared:0.00}");

            Console.WriteLine("=============== End of model evaluation ===============");
        }

        // === Phương thức 4: Dự đoán đơn lẻ ===
        static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("=============== Making a prediction ===============");

            // Tạo Prediction Engine để thực hiện dự đoán
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            // Dữ liệu mẫu: Dự đoán rating cho User ID 6 và Movie ID 10
            var testInput = new MovieRating { userId = 6, movieId = 10 };
            var movieRatingPrediction = predictionEngine.Predict(testInput);

            Console.WriteLine($"\nFor user {testInput.userId} and movie {testInput.movieId}, the predicted rating is: {movieRatingPrediction.Score:0.00}");

            // Đưa ra khuyến nghị dựa trên điểm dự đoán
            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine($"Movie {testInput.movieId} is recommended for user {testInput.userId}");
            }
            else
            {
                Console.WriteLine($"Movie {testInput.movieId} is not recommended for user {testInput.userId}");
            }
        }

        // === Phương thức 5: Lưu Mô hình ===
        static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");
            Console.WriteLine("=============== Saving the model to a file ===============");
            // Lưu mô hình
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
            Console.WriteLine("=============== Model Saved ===============");
        }
    }
}