$neo4jUser = "neo4j"
$neo4jPassword = "your_password"
$neo4jDatabase = "neo4j"

cypher-shell -u $neo4jUser -p $neo4jPassword -d $neo4jDatabase -f database\neo4j\schema\constraints.cypher
cypher-shell -u $neo4jUser -p $neo4jPassword -d $neo4jDatabase -f database\neo4j\schema\indexes.cypher
