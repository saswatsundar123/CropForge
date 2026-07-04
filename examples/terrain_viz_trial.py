"""
CropForge v0.6.0 Deliverable Verification
Terrain Visualisation Trial
"""

from cropforge import Farm, Field, Terrain, RidgeFurrow
from cropforge.plugins import StandardWheat

def main():
    print("Setting up Terrain Viz Trial...")
    farm = Farm(name="Terrain Viz Trial")
    
    # 1. Create a field
    field = Field(name="Hill Furrows", rows=30, cols=30)
    
    # 2. Generate procedural hill topography
    terrain = Terrain.procedural(rows=30, cols=30, generator="undulating", amplitude_m=2.0)
    field.set_terrain(terrain)
    
    # 3. Apply RidgeFurrow land prep modifier
    field.set_land_prep(RidgeFurrow(ridge_height_m=0.3, ridge_spacing_m=1.0))
    
    # 4. Use StandardWheat plugin
    field.use_plugin(StandardWheat)
    
    farm.add_field(field)
    
    # 5. Run for 10 days
    print("Running simulation for 10 days...")
    farm.run(days=10)
    
    # 6. Launch visualization
    print("Launching visualisation server...")
    farm.visualize()

if __name__ == "__main__":
    main()
