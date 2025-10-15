
./toner_tu_sim.py -c banding.yaml -o results/banding --cmap Blues &
./toner_tu_sim.py -c clusters.yaml  -o results/clusters --cmap Reds &
./toner_tu_sim.py -c crystal.yaml -o results/crystal --cmap autumn &
./toner_tu_sim.py -c density_lanes.yaml   -o results/density_lanes --cmap Oranges &
./toner_tu_sim.py -c eddies.yaml -o results/eddies --cmap winter &
./toner_tu_sim.py -c flock.yaml -o results/flock --cmap Purples &
./toner_tu_sim.py -c gas.yaml -o results/gas &
./toner_tu_sim.py -c noise_flocking.yaml -o results/noise_flocking --cmap cool &
./toner_tu_sim.py -c streams.yaml -o results/streams --cmap Greys &
./toner_tu_sim.py -c swirling.yaml  -o results/swirling --cmap Greens &

wait
