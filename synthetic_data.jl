using MLJ, DataFrames, CSV

X,y = make_blobs(10000,2;centers=2)

dfBlobs = DataFrame(X)
dfBlobs.y = y

CSV.write("data/blob10000.txt",dfBlobs)

X,y = make_moons(10000,noise=0.3)

dfMoons = DataFrame(X)
dfMoons.y = y

CSV.write("data/moon10000.txt", dfMoons)







