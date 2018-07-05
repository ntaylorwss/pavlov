# docker hub info
username=ntaylor22
imagecpu=pavlov-cpu
imagegpu=pavlov-gpu

# ensure up to date
version=$(<VERSION)
if ! git diff-index --quiet HEAD -- # true means local changes
then
    echo "You have local changes not on remote. Please commit and push before bumping."
    exit
fi

# get new version
echo "Current version number: $version"
read -p "New version number: " new_version

# bump version, push
echo $new_version > VERSION
git add VERSION
git commit -m "version $new_version"
git tag -a "$new_version" -m "version $new_version"
git push origin master
git push origin master --tags

# rebuild images and bump docker versions
docker build -t ntaylor22/pavlov-gpu:$new_version -t ntaylor22/pavlov-gpu --build-arg baseimage=ntaylor22/tensorflow-gpu . &
docker build -t ntaylor22/pavlov-cpu:$new_version -t ntaylor22/pavlov-gpu --build-arg baseimage=ntaylor22/tensorflow-cpu . &
wait

# push images to hub
docker login -u ntaylor22
docker push ntaylor22/pavlov-gpu:$new_version &
docker push ntaylor22/pavlov-gpu &
docker push ntaylor22/pavlov-cpu:$new_version &
docker push ntaylor22/pavlov-cpu &
wait

# re-run setup and push to pypi
python3 setup.py sdist bdist_wheel
twine upload dist/*
