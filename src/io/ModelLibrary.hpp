#pragma once

#include <memory>
#include <string>
#include <vector>

// A trained model folder as written by the reference 3DGS implementation:
//   <name>/cameras.json
//   <name>/point_cloud/iteration_<N>/point_cloud.ply
struct ModelEntry {
    std::string name;
    std::string plyPath;     // highest iteration's point_cloud.ply
    std::string camerasPath; // empty if the folder has no cameras.json
};

// The trained models under a `data/models` directory. The directory is looked
// up from the working directory and its parents, so the viewer finds it both
// from the repository root and from build/.
class ModelLibrary {
public:
    static std::unique_ptr<ModelLibrary> create();
    bool init();

    // Re-reads the models directory.
    void refresh();

    const std::vector<ModelEntry>& models() const { return m_models; }
    // Empty if no data/models directory was found.
    const std::string& rootPath() const { return m_root; }

private:
    ModelLibrary() = default;

    std::string m_root;
    std::vector<ModelEntry> m_models;
};
