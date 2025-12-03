using System;
using System.IO;


namespace ProductSalesAnomalyDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ProductSalesAnomalyDetection - skeleton");
            Console.WriteLine("1) Put your CSV data into ../data/");
            Console.WriteLine("2) Implement Load -> Build pipeline -> Train -> Detect -> Save model");


            var dataPath = Path.Combine("..", "data", "sales.csv");
            Console.WriteLine($"Data path: {dataPath}");


            // TODO: Add ML.NET code here
            // - Create MLContext
            // - Load data with TextLoader
            // - Use DetectIidSpike/DetectSeasonalSpike or DetectChangePoint
            // - Show results and save model to ../models/
        }
    }
}