using System;
using System.IO;


namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("SentimentAnalysis - skeleton");
            Console.WriteLine("1) Put your CSV data into ../data/ with columns: Label,Text");


            var dataPath = Path.Combine("..", "data", "sentiment-data.csv");
            Console.WriteLine($"Data path: {dataPath}");


            // TODO: Add ML.NET code here
            // - Create MLContext
            // - Load data using LoadFromTextFile
            // - Use FeaturizeText, trainer (SdcaLogisticRegression or others)
            // - Train, Evaluate, Save model
        }
    }
}