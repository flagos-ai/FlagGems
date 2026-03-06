#!/bin/bash

cur_path=$PWD
script_path=$(dirname $0)
# this script help and build whl package
which git 2>/dev/null
have_git=$?
 
cd $script_path
# this part use to get current branch and commit id which use to modify whl package name
if [ $have_git -eq 0 ];then
  git_info=$(git branch -v|grep '^*')
  git_info=${git_info:2}
  branch=$(echo $git_info|awk '{print $1;}')
  commit=$(echo $git_info|awk '{print $2;}')
else
  git_ref=$(cat .git/HEAD|awk '{print $2}')
  branch=$(echo $git_ref|awk -F '/' '{print $NF;}')
  commit=$(cat .git/$git_ref)
  commit=${commit:0:7}
fi
 
# build package
# now patch version to let build different commit version
current_version=$(grep version pyproject.toml |awk -F '"' '{print $(NF-1);}')
new_version="$(echo $current_version|awk -F '.' '{print $1"."$2"."$3}').dev+$commit"
 
# modify whl package name
sed -i "/^version/cversion = \"$new_version\"" pyproject.toml
 
func_check_pip_exit()
{
  echo "Please use those command to for the build requirment"
  echo "pip install build scikit-build-core pybind11 ninja cmake"
  echo "pip install -U setuptools"
  exit
}
 
func_check_pip_ver()
{
  ver=$(pip show $1|grep Version|awk '{print $NF;}'|awk -F '.' '{print $1$2$3}')
  [ $ver -ge $2 ]
}
 
# check build package version
reqirement_miss=0
for i in build scikit-build-core pybind11 ninja cmake setuptools
do
  pip show $i >/dev/null 2>&1
  reqirement_miss=$(($? + $reqirement_miss))
done
 
if [ $reqirement_miss -ne 0 ];then
  func_check_pip_exit
else
  func_check_pip_ver setuptools 6400 # 64.0.0
  reqirement_miss=$(($? + $reqirement_miss))
  func_check_pip_ver scikit-build-core 0110 # 0.11.0
  reqirement_miss=$(($? + $reqirement_miss))
fi
 
if [ $reqirement_miss -ne 0 ];then
  func_check_pip_exit
fi
 
python -m build --no-isolation --wheel .
 
cd $cur_path
