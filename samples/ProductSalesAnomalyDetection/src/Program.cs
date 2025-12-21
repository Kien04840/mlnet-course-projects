using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        string dataPath = Path.Combine(
            Environment.CurrentDirectory,
            "Data",
            "product-sales.csv");

        IDataView dataView =
            mlContext.Data.LoadFromTextFile<ProductSalesData>(
                dataPath,
                hasHeader: true,
                separatorChar: ',');

        int docSize =
            mlContext.Data
                .CreateEnumerable<ProductSalesData>(dataView, false)
                .Count();

        DetectSpike(mlContext, docSize, dataView);
        DetectChangePoint(mlContext, docSize, dataView);
    }


    static void DetectSpike(
      MLContext mlContext,
      int docSize,
      IDataView productSales)
    {
        var iidSpikeEstimator =
            mlContext.Transforms.DetectIidSpike(
                outputColumnName: nameof(ProductSalesPrediction.Prediction),
                inputColumnName: nameof(ProductSalesData.numSales),
                confidence: 95,
                pvalueHistoryLength: docSize / 4);

        ITransformer iidSpikeTransform =
            iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

        IDataView transformedData =
            iidSpikeTransform.Transform(productSales);

        var predictions =
            mlContext.Data.CreateEnumerable<ProductSalesPrediction>(
                transformedData, reuseRowObject: false);

        Console.WriteLine("Spike Detection:");
        foreach (var p in predictions)
        {
            if (p.Prediction == null) continue;

            string result =
                $"{p.Prediction[0]}\t" +
                $"{p.Prediction[1]:F2}\t" +
                $"{p.Prediction[2]:E2}";

            if (p.Prediction[0] == 1)
                result += "  <-- Spike detected";

            Console.WriteLine(result);
        }

        Console.WriteLine();
    }


    static void DetectChangePoint(
     MLContext mlContext,
     int docSize,
     IDataView productSales)
    {
        Console.WriteLine("Detect Persistent changes in pattern");
        Console.WriteLine("=============== Training the model Using Change Point Detection Algorithm===============");
        Console.WriteLine("=============== End of training process ===============");

        var iidChangePointEstimator =
            mlContext.Transforms.DetectIidChangePoint(
                outputColumnName: nameof(ProductSalesPrediction.Prediction),
                inputColumnName: nameof(ProductSalesData.numSales),
                confidence: 95,
                changeHistoryLength: docSize / 4);

        ITransformer iidChangePointTransform =
            iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

        IDataView transformedData =
            iidChangePointTransform.Transform(productSales);

        var predictions =
            mlContext.Data.CreateEnumerable<ProductSalesPrediction>(
                transformedData, reuseRowObject: false);

        Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

        foreach (var p in predictions)
        {
            if (p.Prediction is null) continue;

            string result =
                $"{p.Prediction[0]}\t" +
                $"{p.Prediction[1]:F2}\t" +
                $"{p.Prediction[2]:F2}\t" +
                $"{p.Prediction[3]:F2}";

            if (p.Prediction[0] == 1)
                result += "  <-- alert is on, predicted changepoint";

            Console.WriteLine(result);
        }

        Console.WriteLine();
    }


    static IDataView CreateEmptyDataView(MLContext mlContext) =>
        mlContext.Data.LoadFromEnumerable(new List<ProductSalesData>());
}
