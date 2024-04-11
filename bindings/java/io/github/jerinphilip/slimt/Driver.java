package io.github.jerinphilip.slimt;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

class Driver {
  public static void main(String[] args) {
    int encoderLayers = 6;
    int decoderLayers = 2;
    int feedForwardDepth = 2;
    int numHeads = 8;
    ModelConfig config =
        new ModelConfig(encoderLayers, decoderLayers, feedForwardDepth, numHeads, "paragraph");
    // Package archive = new Package();
    String root = args[0];

    int cacheSize = 1024;
    Service service = new Service(cacheSize);

    Package archive =
        new Package(
            Paths.get(root, args[1]).toString(),
            Paths.get(root, args[2]).toString(),
            Paths.get(root, args[3]).toString(),
            "");

    Model model = new Model(config, archive);
    System.out.println("Construction success");
    boolean html = false;
    List<String> sources = new ArrayList<>();
    sources.add("Hello World. Help me out here, will you?");
    sources.add("Goodbye World. Fine, don't help me.");
    String[] targets = service.translate(model, sources, html);
    for (int i = 0; i < sources.size(); i++) {
      System.out.println("> " + sources.get(i));
      System.out.println("< " + targets[i]);
    }
  }
}
