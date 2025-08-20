#!/bin/sh -x


f=$(basename "$1")
f="${f%.*}"

lat=$(gdalinfo $f-elevation.tif | grep LATITUDE | cut -d = -f 2)

~/farsite/src/lcpmake -latitude $lat \
    -landscape  $f \
    -elevation  $f-elevation.asc \
    -slope      $f-slope.asc \
    -aspect     $f-aspect.asc \
    -fuel       $f-fuel.asc \
    -cover      $f-cc.asc \
    -height     $f-ch.asc \
    -base       $f-cbh.asc \
    -density    $f-cbd.asc
