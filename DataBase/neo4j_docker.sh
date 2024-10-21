docker run -d -p 7474:7474 -p 7687:7687 -v ${PWD}:/data -e NEO4J_AUTH=none --name some-neo4j neo4j
docker exec -it some-neo4j cypher-shell
