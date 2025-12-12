using System;
using System.IO;


namespace TransferLearningTF
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("TransferLearningTF - skeleton");
            Console.WriteLine("1) Prepare images in ../data/train/<label>/*.jpg and ../data/test/<label>/*.jpg");
            Console.WriteLine("2) Put pretrained TF model (if needed) into ../models/");


            var trainFolder = Path.Combine("..", "data", "train");
            Console.WriteLine($"Train folder: {trainFolder}");


            // TODO: Add ML.NET image classification pipeline
            // - Load images with ImageLoadingEstimator
            // - Resize/transform
            // - Use TensorFlow or ImageClassificationTrainer for transfer learning
            // - Train and save the model
        }
    }
}