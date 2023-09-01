using SFML.Graphics;
using SFML.System;
using SFML.Window;
using Color = SFML.Graphics.Color;

namespace NeuralNetworkExample.ExampleImplement.DataSection.Implement
{
    public class DataNNSectionWithSFML
    {
        public Vertex[] mapPoints;
        public List<Vertex> myPoints = new List<Vertex>();
        public readonly NeuralNetwork neuralNetwork;

        public readonly double _width;
        public readonly double _height;

        public DataNNSectionWithSFML(int width, int height, NeuralNetwork? neuralNetwork = null)
        {
            _width = width;
            _height = height;
            if (neuralNetwork == null)
            {
                this.neuralNetwork = new NeuralNetwork(2, new[] { 5 }, 2, 0.01, 0.10m);
            }
            else if (neuralNetwork.layers.First().GetLength(0) != 3 || neuralNetwork.layers.Last().GetLength(0) != 2)
            {
                this.neuralNetwork = new NeuralNetwork(2, new[] { 5 }, 2, 0.01, 0.10m);
            }
            else
            {
                this.neuralNetwork = neuralNetwork;
            }
        }

        public void Visualize()
        {
            RenderWindow window = new RenderWindow(new VideoMode((uint)_width, (uint)_height), "sfml");
            window.SetVerticalSyncEnabled(true);
            mapPoints = new Vertex[(int)(_width * _height)];

            for (int i = 0; i < _height; i++)
            {
                for (int k = 0; k < _width; k++)
                {
                    mapPoints[i * (int)_width + k] = new Vertex(new Vector2f(k, i), Color.Green);
                }
            }

            bool rc = false;
            bool lc = false;

            window.Closed += (object? sender, EventArgs e) => window.Close();
            window.MouseMoved += (object? sender, MouseMoveEventArgs e) =>
            {
                if (lc) { myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Green)); }
                if (rc) { myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Blue)); }
            };

            window.MouseButtonPressed += (object? sender, MouseButtonEventArgs e) =>
            {
                switch (e.Button)
                {
                    case Mouse.Button.Left:
                        myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Green));
                        lc = true;
                        break;
                    case Mouse.Button.Right:
                        rc = true;
                        myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Blue));
                        break;
                }
            };

            window.MouseButtonReleased += (object? sender, MouseButtonEventArgs e) =>
            {
                switch (e.Button)
                {
                    case Mouse.Button.Left:
                        myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Green));
                        lc = false;
                        break;
                    case Mouse.Button.Right:
                        rc = false;
                        myPoints.Add(new Vertex(new Vector2f(e.X, e.Y), Color.Blue));
                        break;
                    case Mouse.Button.Middle:
                        break;
                    case Mouse.Button.XButton1:
                        break;
                    case Mouse.Button.XButton2:
                        break;
                    case Mouse.Button.ButtonCount:
                        break;
                    default:
                        break;
                }
            };

            while (window.IsOpen)
            {
                window.DispatchEvents();
                for (int i = 0; i < myPoints.Count; i++)
                {
                    var point = myPoints[i];
                    neuralNetwork.inputSignalInlayer(neuralNetwork.layers.First(), point.Position.X / _width, point.Position.Y / _height);
                    neuralNetwork.feedForwards();

                    var expect = new double[] { point.Color == Color.Green ? 1 : 0, point.Color == Color.Blue ? 1 : 0 };

                    neuralNetwork.findError(neuralNetwork.layers.Last(), expect);
                    neuralNetwork.findErrors();
                    neuralNetwork.fixWeight();
                }

                for (int i = 0; i < mapPoints.Length; i++)
                {
                    var point = mapPoints[i];
                    neuralNetwork.inputSignalInlayer(neuralNetwork.layers.First(), point.Position.X / _width, point.Position.Y / _height);
                    neuralNetwork.feedForwards();

                    mapPoints[i].Color = neuralNetwork.layers.Last()[0, 0] > neuralNetwork.layers.Last()[1, 0] ? new Color(100, 255, 0) : new Color(100, 0, 255);
                }

                window.Draw(mapPoints.ToArray(), PrimitiveType.Points);
                window.Draw(myPoints.ToArray(), PrimitiveType.Points);


                window.Display();
            }
        }
    }
}
