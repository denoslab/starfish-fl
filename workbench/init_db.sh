#!/bin/bash
# Initialize the Starfish router database

docker exec -i starfish-postgres \
psql -U postgres << EOF
CREATE DATABASE "starfish-router";
EOF

