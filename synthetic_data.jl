using MLJ, DataFrames, CSV

X,y = make_blobs(1000,2;centers=2)

dfBlobs = DataFrame(X)
dfBlobs.y = y

CSV.write("data/blob1000.txt",dfBlobs)

X,y = make_moons(1000,noise=0.3)

dfMoons = DataFrame(X)
dfMoons.y = y

CSV.write("data/moon1000.txt", dfMoons)







