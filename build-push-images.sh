docker build -t hexagon-base -f Dockerfile.base .
docker tag hexagon-base aliostad/hexagon-base
docker push aliostad/hexagon-base
docker build -t hexagon-server -f Dockerfile.server .
docker build -t hexagon-player -f Dockerfile.player .
docker tag hexagon-player aliostad/hexagon-player
docker tag hexagon-server aliostad/hexagon-server
docker push aliostad/hexagon-player
docker push aliostad/hexagon-server
