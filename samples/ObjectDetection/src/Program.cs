using System;
using System.IO;


namespace ObjectDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ObjectDetection - skeleton");
            Console.WriteLine("1) Put ONNX model tiny-yolov2.onnx into ../models/");
            Console.WriteLine("2) Put test images into ../data/");


            var modelPath = Path.Combine("..", "models", "tinyyolov2.onnx");
            Console.WriteLine($"Model path: {modelPath}");


            // TODO: Add ML.NET ONNX inference code
            // - Load image(s)
            // - ApplyOnnxModel to score
            // - Post-process outputs into bounding boxes and classes
            // - Save results / visualize
        }
    }
}