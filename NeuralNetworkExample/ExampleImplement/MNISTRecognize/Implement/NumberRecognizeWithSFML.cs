using SFML.Graphics;
using SFML.System;
using SFML.Window;
using System.Drawing;
using Color = SFML.Graphics.Color;
using Font = SFML.Graphics.Font;

namespace NeuralNetworkExample.ExampleImplement.MNISTRecognize.Implement
{
    public class NumberRecognizeWithSFML
    {
        public Vertex[] mapPoints;
        public List<Vertex> myPoints = new List<Vertex>();
        public readonly NeuralNetwork neuralNetwork;

        public readonly int _widthPixel = 28;
        public readonly int _heightPixel = 28;
        public readonly int scalePixel = 10;

        public readonly int _sizeForHud = 150;

        public readonly double _width;
        public readonly double _height;


        private static Color _backColor = Color.Black;
        private static Color _NumberColor = Color.White;
        private static Color _HudColor = Color.White;
        private static Color _BarsColor = Color.Green;

        private const string _defaultSaveNN = "DefaultNN";



        private NeuralNetwork _getDefaultNN() => new NeuralNetwork(784, new[] { 512, 128, 32 }, 10, 0.01, 0.01m);

        public NumberRecognizeWithSFML(NeuralNetwork? neuralNetwork = null)
        {
            if (neuralNetwork == null)
            {
                this.neuralNetwork = _getDefaultNN();
            }
            else if (neuralNetwork.layers.First().GetLength(0) != 784+1 || neuralNetwork.layers.Last().GetLength(0) != 10)
            {
                this.neuralNetwork = _getDefaultNN();
            }
            else
            {
                this.neuralNetwork = neuralNetwork;
            }

            _width = _widthPixel * scalePixel;
            _height = _heightPixel * scalePixel;
        }

        public void LearnNNRecognize(string path, bool saveNN = false)
        {
            var files=Directory.GetFiles(path);
            decimal error = neuralNetwork.errorTolerance + 100m;

            while (error>neuralNetwork.errorTolerance)
            {
                foreach (var file in files)
                {
                    _ = int.TryParse(Path.GetFileName(file)[Path.GetFileName(file).IndexOf('.') - 1].ToString(), out int expectedNum);

                    var bitmap = new Bitmap(file);
                    List<double> signals = new List<double>();
                    for (int i = 0; i < _heightPixel; i++)
                    {
                        for (int k = 0; k < _widthPixel; k++)
                        {
                            signals.Add(bitmap.GetPixel(k, i).R / 255.0);
                        }
                    }

                    neuralNetwork.inputSignalInlayer(neuralNetwork.layers.First(), signals.ToArray());
                    neuralNetwork.feedForwards();

                    var expected = new double[10];
                    expected[expectedNum] = 1;

                    error=neuralNetwork.findError(neuralNetwork.layers.Last(), expected);
                    neuralNetwork.findErrors();
                    neuralNetwork.fixWeight();
                }

                if (saveNN)
                {
                    neuralNetwork.saveNN(_defaultSaveNN);
                }
            }
        }

        public void Visualize()
        {
            RenderWindow window = new RenderWindow(new VideoMode((uint)(_width + _sizeForHud), (uint)_height), "sfml");
            window.SetVerticalSyncEnabled(true);

            mapPoints = new Vertex[_widthPixel * _heightPixel];
            for (int i = 0; i < _heightPixel; i++)
            {
                for (int k = 0; k < _widthPixel; k++)
                {
                    mapPoints[i * _widthPixel + k] = new Vertex(new Vector2f(k, i), _backColor);
                }
            }

            bool rc = false;
            bool lc = false;

            window.Closed += (object? sender, EventArgs e) => window.Close();

            window.MouseMoved += (object? sender, MouseMoveEventArgs e) =>
            {
                if ((e.Y >= _width || e.Y <= 0) || (e.X >= _width || e.X <= 0))
                {
                    return;
                }
                if (lc) { mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _NumberColor; }
                if (rc) { mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _backColor; }
            };

            window.MouseButtonPressed += (object? sender, MouseButtonEventArgs e) =>
            {
                if ((e.Y >= _width || e.Y <= 0) || (e.X >= _width || e.X <= 0))
                {
                    return;
                }
                switch (e.Button)
                {
                    case Mouse.Button.Left:
                        mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _NumberColor;
                        lc = true;
                        break;
                    case Mouse.Button.Right:
                        rc = true;
                        mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _backColor;
                        break;
                    case Mouse.Button.Middle:
                        for (int i = 0; i < mapPoints.Length; i++)
                        {
                            mapPoints[i].Color = _backColor;
                        }
                        break;
                }
            };

            window.MouseButtonReleased += (object? sender, MouseButtonEventArgs e) =>
            {
                if ((e.Y >= _width || e.Y <= 0) || (e.X >= _width || e.X <= 0))
                {
                    return;
                }
                switch (e.Button)
                {
                    case Mouse.Button.Left:
                        mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _NumberColor;
                        lc = false;
                        break;
                    case Mouse.Button.Right:
                        rc = false;
                        mapPoints[e.Y / scalePixel * _widthPixel + e.X / scalePixel].Color = _backColor;
                        break;
                }
            };

            RectangleShape[] pixels = new RectangleShape[mapPoints.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = new RectangleShape(new Vector2f(scalePixel, scalePixel));
            }

            Vertex[] line = new Vertex[]
            {
                new Vertex(new((float)_width,0),_HudColor),
                new Vertex(new((float)_width,(float)_height),_HudColor)
            };

            Text[] numbers = new Text[10];
            for (int i = 0; i < numbers.Length; i++)
            {
                numbers[i]= new Text(i.ToString(), new Font("defaultFont.ttf"));
                numbers[i].FillColor = _backColor;
                numbers[i].OutlineThickness = 1;
                numbers[i].OutlineColor = _HudColor;
                numbers[i].Position = new Vector2f((float)(_width + scalePixel), i*27);
            }

            RectangleShape[] numberBars= new RectangleShape[10];
            for (int i = 0; i < numbers.Length; i++)
            {
                numberBars[i] = new RectangleShape(new Vector2f(100,20));
                numberBars[i].FillColor = _HudColor;
                numberBars[i].FillColor = _backColor;
                numberBars[i].OutlineThickness = 1;
                numberBars[i].Position = new Vector2f(320, i * 27+12);
            }

            RectangleShape[] numberFillBars = new RectangleShape[10];
            for (int i = 0; i < numbers.Length; i++)
            {
                numberFillBars[i] = new RectangleShape(new Vector2f(100, 20));
                numberFillBars[i].FillColor = _BarsColor;
                numberFillBars[i].Position = new Vector2f(320, i * 27 + 12);
            }

            while (window.IsOpen)
            {
                window.DispatchEvents();

                for (int i = 0; i < mapPoints.Length; i++)
                {
                    pixels[i].FillColor = mapPoints[i].Color;
                    pixels[i].Position = new Vector2f(mapPoints[i].Position.X * scalePixel, mapPoints[i].Position.Y * scalePixel);
                }

                neuralNetwork.inputSignalInlayer(neuralNetwork.layers.First(), mapPoints.Select(x=>x.Color==_NumberColor?1.0:0.0).ToArray());   
                neuralNetwork.feedForwards();
                for (int i = 0; i < numberFillBars.Length; i++)
                {
                    int sizeX = (int)(neuralNetwork.layers.Last()[i, 0] * 100);
                    numberFillBars[i].Size = new Vector2f(sizeX, 20);
                }

                window.Clear();
                foreach (var item in pixels)
                {
                    window.Draw(item);
                }
                foreach (var item in numbers)
                {
                    window.Draw(item);
                }
                foreach (var item in numberBars)
                {
                    window.Draw(item);
                }
                foreach (var item in numberFillBars)
                {
                    window.Draw(item);
                }

                window.Draw(line, PrimitiveType.Lines);
                window.Display();
            }
        }
    }
}
