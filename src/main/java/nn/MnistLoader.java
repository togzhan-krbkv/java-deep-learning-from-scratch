package nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * MnistLoader, reads MNIST from CSV in Kaggle's format:
 *
 * <pre>
 *   label,pixel0,pixel1,...,pixel783
 * </pre>
 *
 * Pixel values are normalized to [0, 1]; labels become one-hot vectors
 * of length 10, ready to feed into a NeuralNetwork.
 *
 * @author Togzhan K.
 */
public class MnistLoader
{
    public static final int IMAGE_SIZE = 784;
    public static final int NUM_CLASSES = 10;

    /** input: 784x1 normalized pixels, target: 10x1 one-hot, label: raw digit 0-9 */
    public static class Example
    {
        public final Matrix input;
        public final Matrix target;
        public final int label;

        public Example(Matrix input, Matrix target, int label)
        {
            this.input = input;
            this.target = target;
            this.label = label;
        }
    }

    /** Loads examples from a CSV file; set skipHeader true if the first line is a header */
    public static List<Example> load(String path, boolean skipHeader) throws IOException
    {
        List<Example> examples = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path)))
        {
            String line = reader.readLine();
            if (line != null && skipHeader)
            {
                line = reader.readLine();
            }

            while (line != null)
            {
                if (!line.isEmpty())
                {
                    examples.add(parseLine(line));
                }
                line = reader.readLine();
            }
        }

        return examples;
    }

    private static Example parseLine(String line)
    {
        String[] parts = line.split(",");
        if (parts.length != IMAGE_SIZE + 1)
        {
            throw new IllegalArgumentException(
                    "Expected " + (IMAGE_SIZE + 1) + " values per row, got " + parts.length);
        }

        int label = Integer.parseInt(parts[0].trim());
        if (label < 0 || label >= NUM_CLASSES)
        {
            throw new IllegalArgumentException("Label out of range: " + label);
        }

        double[] pixels = new double[IMAGE_SIZE];
        for (int i = 0; i < IMAGE_SIZE; i++)
        {
            pixels[i] = Integer.parseInt(parts[i + 1].trim()) / 255.0;
        }
        Matrix input = Matrix.columnVector(pixels);

        double[] oneHot = new double[NUM_CLASSES];
        oneHot[label] = 1.0;
        Matrix target = Matrix.columnVector(oneHot);

        return new Example(input, target, label);
    }
}
