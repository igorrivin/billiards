#!/bin/bash
# Sample workflow for finding and analyzing billiard orbits

echo "=== Finding periodic orbits for p=3, N=5 ==="
python ../orbit_finder.py 3.0 5 1000

echo -e "\n=== Examining results ==="
echo "First few orbits found:"
head -5 p3.0_N5_orbits.csv

echo -e "\n=== Verifying the first orbit ==="
# Extract first orbit (skip header)
THETA=$(sed -n '2p' p3.0_N5_orbits.csv | cut -d'"' -f2)
echo "Theta values: $THETA"
python ../verify_orbit.py 3.0 "$THETA" --verbose

echo -e "\n=== Plotting some interesting orbits ==="
# Plot orbits with different signatures
python ../plot_orbit.py --csv p3.0_N5_orbits.csv --rows 1,10,20 --grid --save orbits_grid.png

echo -e "\n=== Summary statistics ==="
python -c "
import json
with open('p3.0_N5_metadata.json') as f:
    data = json.load(f)
print('Total orbits found:', data['overall']['total_orbits'])
print('Signatures found:', list(data['orbit_counts_by_signature'].keys()))
print('Orbit counts by signature:')
for sig, count in data['orbit_counts_by_signature'].items():
    print(f'  {sig}: {count} orbits')
"

echo -e "\nWorkflow complete! Check orbits_grid.png for visualizations."