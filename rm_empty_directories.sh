#!/bin/bash

#EMPTY_DIRECTORIES=$(find -type d -empty)
#for DIRECTORY in "$EMPTY_DIRECTORIES"; do
#while IFS= read -d '' $DIRECTORY; do
function safe_rmdir() {
	local DIRECTORY=$1;

	if [[ "$DIRECTORY" == *"git"* ]]; then
		# Git directory, should not touch
#		echo "Will not remove $DIRECTORY";
		return;
	fi

	rmdir "$DIRECTORY";
}

export -f safe_rmdir;
find -type d -empty -exec bash -c 'safe_rmdir "$1"' _ {} \;