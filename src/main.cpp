#include "app/OfflineRenderer.hpp"
#include "app/Options.hpp"
#include "app/Viewer.hpp"
#include "io/CameraSet.hpp"
#include "scene/GaussianCloud.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        std::unique_ptr<Options> options = Options::create(argc, argv);
        if (!options || options->help) {
            Options::printUsage(argv[0]);
            return options ? 0 : 1;
        }

        // The viewer loads the scene itself (in the background, with progress).
        std::unique_ptr<GaussianCloud> cloud;
        if (!options->outputDir.empty() && !options->scenePath.empty()) {
            cloud = GaussianCloud::createFromPly(options->scenePath);
            if (!cloud) return 1;
            std::cout << "Loaded " << cloud->size() << " Gaussians (SH degree " << cloud->shDegree << ") from "
                      << options->scenePath << "\n";
        }

        std::unique_ptr<CameraSet> cameras;
        if (!options->camerasPath.empty()) {
            cameras = CameraSet::create(options->camerasPath);
            if (!cameras) return 1;
            std::cout << "Loaded " << cameras->size() << " cameras from " << options->camerasPath << "\n";
        }

        if (!options->outputDir.empty()) {
            std::unique_ptr<OfflineRenderer> offline = OfflineRenderer::create(*options, *cloud, cameras.get());
            return offline && offline->run() ? 0 : 1;
        }
        std::unique_ptr<Viewer> viewer = Viewer::create(*options, std::move(cameras));
        return viewer && viewer->run() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
