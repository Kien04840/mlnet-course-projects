using System;
using System.IO;


namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("MovieRecommender - skeleton");
            Console.WriteLine("1) Put ratings data into ../data/ratings.csv (userId,movieId,rating)");


            var dataPath = Path.Combine("..", "data", "ratings.csv");
            Console.WriteLine($"Data path: {dataPath}");


            // TODO: Add ML.NET code here
            // - Create MLContext
            // - Load ratings dataset
            // - Use MatrixFactorizationTrainer for recommender
            // - Train, Save model, and sample PredictRating
        }
    }
}