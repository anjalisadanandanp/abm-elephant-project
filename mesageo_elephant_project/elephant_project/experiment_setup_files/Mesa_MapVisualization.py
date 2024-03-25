from mesa_geo.visualization.MapModule import MapModule

"""
Overwriting function:render in MapModule
Portrayal of Human agent, Elephant agents and food 

"""

class MapModule(MapModule):
    """A MapModule for Leaflet maps.""" 

    def render(self, model):

        featurecollection = dict(type="FeatureCollection", features=[])

        for agent in model.grid.agents:

            agent_vars = agent.__geo_interface__()   #agent_vars is a dictionary: (['type', 'geometry', 'properties'])
            portrayal = self.portrayal_method(agent)    #The portrayal dict defined in server.py

            agent_portrayal = {}
            agent_portrayal["type"] = agent_vars["type"]
            agent_portrayal["geometry"] = agent_vars["geometry"]
            agent_portrayal["properties"] = {}      #we are overwriting properties for rendering the agents

            for key, value in portrayal.items():
                agent_portrayal["properties"][key] = value

            if "bull" in agent.unique_id:
                agent_portrayal["properties"]["category"] = "Elephant"
                agent_portrayal["properties"]["color"] = "red"
                agent_portrayal["properties"]["radius"] = 5

            else:
                agent_portrayal["properties"]["category"] = "Human"
                agent_portrayal["properties"]["color"] = "black"
                agent_portrayal["properties"]["radius"] = 2

            if agent.fitness <= 0:
                agent_portrayal["properties"]["color"] = "grey"
                agent_portrayal["properties"]["radius"] = 1

            featurecollection["features"].append(agent_portrayal)

        return featurecollection
