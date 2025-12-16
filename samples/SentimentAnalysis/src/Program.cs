using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

using System;
using System.IO;

public class Program
{
    // Đường dẫn đến tệp dữ liệu yelp_labelled.txt. 
    // Yêu cầu tệp này phải nằm trong thư mục Data của thư mục đầu ra (bin/Debug/netX.X).
    static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

    static void Main(string[] args)
    {
        // 1. Khởi tạo MLContext
        MLContext mlContext = new MLContext();

        // 2. Tải và Chia dữ liệu
        TrainTestData splitDataView = LoadData(mlContext);

        // 3. Xây dựng và Huấn luyện mô hình
        ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

        // 4. Đánh giá mô hình
        Evaluate(mlContext, model, splitDataView.TestSet);

        // 5. Sử dụng mô hình để dự đoán đơn lẻ
        UseModelForSinglePrediction(mlContext, model);
    }

    // Phương thức tải dữ liệu và chia thành tập huấn luyện và tập kiểm thử
    public static TrainTestData LoadData(MLContext mlContext)
    {
        Console.WriteLine("=============== Loading Data ===============");
        // Tải dữ liệu từ tệp text
        IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

        // Chia dữ liệu thành 80% Training (huấn luyện) và 20% Test (kiểm thử)
        TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        Console.WriteLine("=============== End of loading Data ===============");
        Console.WriteLine();

        return splitDataView;
    }

    // Phương thức xây dựng pipeline và huấn luyện mô hình
    public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    {
        // Định nghĩa pipeline chuyển đổi dữ liệu và thuật toán huấn luyện
        var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            // Sử dụng thuật toán phân loại nhị phân SdcaLogisticRegression
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

        Console.WriteLine("=============== Create and Train the Model ===============");

        // Huấn luyện mô hình
        var model = estimator.Fit(splitTrainSet);

        Console.WriteLine("=============== End of training ===============");
        Console.WriteLine();

        return model;
    }

    // Phương thức đánh giá hiệu suất của mô hình
    public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    {
        Console.WriteLine("=============== Evaluating Model accuracy ===============");

        // Áp dụng mô hình lên tập dữ liệu kiểm thử
        IDataView predictions = model.Transform(splitTestSet);

        // Đánh giá và lấy các chỉ số hiệu suất
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"Area Under ROC Curve (AUC): {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        Console.WriteLine("=============== End of Model Evaluation ===============");
        Console.WriteLine();
    }

    // Phương thức thực hiện dự đoán trên một mẫu dữ liệu đơn lẻ
    private static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
    {
        Console.WriteLine("=============== Test a single prediction ===============");

        // Tạo PredictionEngine để dự đoán một mẫu đầu vào
        var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        // Mẫu 1: Bình luận tiêu cực
        var sampleStatement1 = new SentimentData { SentimentText = "This was a horrible and unpleasant experience" };
        var resultPrediction1 = predictionFunction.Predict(sampleStatement1);
        Console.WriteLine($"Sentiment: {sampleStatement1.SentimentText} | Prediction: {(resultPrediction1.Prediction ? "Positive" : "Negative")} | Probability: {resultPrediction1.Probability:P2} | Score: {resultPrediction1.Score}");

        // Mẫu 2: Bình luận tích cực
        var sampleStatement2 = new SentimentData { SentimentText = "This is a great place to eat and I love it" };
        var resultPrediction2 = predictionFunction.Predict(sampleStatement2);
        Console.WriteLine($"Sentiment: {sampleStatement2.SentimentText} | Prediction: {(resultPrediction2.Prediction ? "Positive" : "Negative")} | Probability: {resultPrediction2.Probability:P2} | Score: {resultPrediction2.Score}");

        Console.WriteLine("=============== End of single prediction test ===============");
        Console.WriteLine();
    }
}

// Định nghĩa lớp dữ liệu đầu vào (Data In/Load)
public class SentimentData
{
    // LoadColumn(0) là cột đầu tiên trong tệp dữ liệu (SentimentText)
    [LoadColumn(0)]
    public string? SentimentText;

    // LoadColumn(1) là cột thứ hai (Sentiment, True=Positive, False=Negative), được gán nhãn là "Label"
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment;
}

// Định nghĩa lớp dữ liệu đầu ra (Prediction Out)
public class SentimentPrediction : SentimentData
{
    // PredictedLabel là nhãn được mô hình dự đoán
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    // Probability: Xác suất của dự đoán
    public float Probability { get; set; }

    // Score: Điểm thô (raw score) của dự đoán
    public float Score { get; set; }
}