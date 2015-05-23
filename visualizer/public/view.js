(function (d3, websites, undefined) {
    document.addEventListener('DOMContentLoaded', function () {
        window.viewer = new Viewer();
        window.watcher = new Watcher(window.viewer);
    });

    //
    // Manage Input
    // * Read values, Send requires, Update status, Update graph
    //
    function Watcher(viewer) {
        if (!(this instanceof Watcher)) return new Watcher();
        var self = this;

        this._viewer = viewer;

        this._input = document.querySelector('#input input[type=text]');

        this._input.addEventListener('input', this._abort.bind(this));
        this._input.addEventListener('keyup', function (event) {
            if (event.keyIdentifier === "Enter") self._search();
        });
    }

    Watcher.prototype._abort = function () {
        if (this._xhr) this._xhr.abort();
        this._viewer.status('none');
    };

    Watcher.prototype._search = function () {
        var self = this;

        var query = null;
        var method = this._input.value.trim().slice(0, 6);
        var text = this._input.value.trim().slice(6).trim();
        switch (method) {
            case 'title:': query = this._title(text); break;
            case 'group:': query = this._group(text); break;
        }

        if (query === null) {
            return this._viewer.status('error', 'Unkown command');
        } else if (query instanceof Error) {
            return this._viewer.status('error', query.message);
        } else {
            this._viewer.status('wait', 'Loading ...');

            this._xhr = d3.json('/graph.json?' + query, function (error, data) {
                if (error) return self._viewer.status('error', 'Request error: ' + error.statusText);

                self._viewer.status('none');
                self._viewer.update(data);
          });
        }
    };

    Watcher.prototype._group = function (text) {
        // Create array of groups
        var groups = text
            .split(/\s*,\s*/)
            .map(function (str) { return parseInt(str, 10); });

        // If there where values parseInt didn't understand tell the viewer
        if (groups.filter(isNaN).length > 0) {
            return new Error('Did not understand group input');
        } else {
            return "groups=" + encodeURIComponent(groups.join(','));
        }
    };

    Watcher.prototype._title = function (text) {
        return "title=" + encodeURIComponent(text);
    };

    //
    // Manage the view
    //
    function Viewer() {
        if (!(this instanceof Viewer)) return new Viewer();

        this._graph = new GraphViewer();
        this._status = document.querySelector('#status');
        this._list = new ListViewer();
        this._details = new DetailsView(this);
        this._handler = new NodeEventHandler(this);

        this._data = new DataContainer([], {
            _group2index: {}
        });
    }

    Viewer.prototype.status = function (key, message) {
        var colormap = {
            'error': 'rgb(172, 65, 35)',
            'wait': 'rgb(121, 132, 18)',
            'none': 'rgb(255, 255, 255)'
        };

        this._status.innerHTML = message || '';
        this._status.style.color = colormap[key];
    };

    Viewer.prototype.update = function (data) {
        this._data = new DataContainer(data, this._data);

        this._list.update(this._data);
        this._graph.update(this._data);
    };

    //
    // Manage events on nodes
    //
    function NodeEventHandler(watcher) {
        this._watcher = watcher;
        this._detailsViewer = this._watcher._details;
        this._listViewer = this._watcher._list;
        this._graphViewer = this._watcher._graph;

        this._leftHighlight = null;
        this._rightHighlight = null;

        this.addEventListener('click', this._click.bind(this));
    }

    NodeEventHandler.prototype.addEventListener = function (eventName, fn) {
        var self = this;
        function handler(d, i) {
            var list = self._listViewer.getNode(d);
            var graph = self._graphViewer.getNode(d);
            fn(d, list, graph);
        }

        this._listViewer.addEventListener(eventName, handler);
        this._graphViewer.addEventListener(eventName, handler);
    };

    NodeEventHandler.prototype._click = function (d, listNode, graphNode) {
        var side = (d3.event.altKey ? 'right' : 'left');
        var highlightName = '_' + side + 'Highlight';

        if (this[highlightName]) {
            this[highlightName].list.classList.remove('highlight');
            this[highlightName].graph.classList.remove('highlight');
        }
        listNode.classList.add('highlight');
        graphNode.classList.add('highlight');

        this[highlightName] = { 'list': listNode, 'graph': graphNode };
        if (!isElementVisible(listNode)) listNode.scrollIntoView(true);

        this._detailsViewer[side](d);
    };

    //
    // Holds and unpack the data
    //
    function LinkContainer(source, target, distance) {
        this.source = source;
        this.target = target;
        this.distance = distance;
    }

    function NodeContainer(data) {
        this.title = data[0];
        this.website = data[1];
        this.id = data[2];
    }

    function GroupContainer(data) {
        this._id2index = {};

        this.group = data.group;
        this.nodes = [];
        this.links = [];

        var i = 0;

        // Create node list
        for (i=0; i < data.nodes.length; i++) {
            var node = data.nodes[i];
            this._id2index[node[2]] = i;
            this.nodes.push(new NodeContainer(node));
        }

        // Create link list and find min max for distance
        var min = Infinity, max = 0;
        for (i=0; i < data.links.length; i++) {
            var link = data.links[i];

            this.links.push(new LinkContainer(
                this.nodes[ this._id2index[link[0]] ],
                this.nodes[ this._id2index[link[1]] ],
                link[2]
            ));

            if (link[2] < min) min = link[2];
            if (link[2] > max) max = link[2];
        }

        // Save min & max
        this.min = min;
        this.max = max;
    }

    function DataContainer(data, old) {
        if (!(this instanceof DataContainer)) return new DataContainer(data);

        this.groups = [];
        this._group2index = {};

        // Create group list and find min max for distance
        var min = Infinity, max = 0;
        for (var i = 0; i < data.length; i++) {
            var groupId = data[i].group;

            // Get group, either from an old build or create it
            var group;
            if (old._group2index.hasOwnProperty(groupId)) {
                group = old.groups[ old._group2index[groupId] ];
            } else {
                group = new GroupContainer(data[i]);
            }

            this.groups.push(group);
            this._group2index[group.group] = i;

            if (group.min < min) min = group.min;
            if (group.max > max) max = group.max;
        }

        // Save min & max
        this.min = min;
        this.max = max;
    }

    //
    // Manage the Node View
    //
    function ListViewer() {
        if (!(this instanceof ListViewer)) return ListViewer();
        EventAbstraction.apply(this, arguments);

        this._view = d3.select('#nodes');
        this._colors = d3.scale.category10();
        this._groupSelection = this._view.selectAll('.group');
    }
    inherits(ListViewer, EventAbstraction);

    ListViewer.prototype.update = function (data) {
        var self = this;

        this._groupSelection = this._groupSelection
            .data(data.groups, function (d) { return d.group; });

        var group = this._groupSelection.enter().append('div')
            .attr('class', 'group')
            .text(function (d) {  return d.group; });

        var nodes = group.selectAll('.node')
            .data(function (d) { return d.nodes; }, function (d) { return d.id; });

        var node = nodes.enter().append('div')
            .attr('class', 'node')
            .call(this.enterNodeEventer.bind(this))
            .insert('span', ':first-child')
                .attr('class', 'nodecolor')
                .style('color', function (d) { return self._colors(d.website); })
                .each(function (d) {
                    this.parentNode.appendChild(document.createTextNode(d.title));
                });

        this._groupSelection.exit().remove();
        nodes.exit()
            .call(this.exitNodeEventer.bind(this))
            .remove();
    };

    //
    // Manage the Graph
    //
    function GraphViewer() {
        if (!(this instanceof GraphViewer)) return GraphViewer();
        EventAbstraction.apply(this, arguments);
        var self = this;

        this._svg = d3.select('#graph');

        this._colors = d3.scale.category10();
        this._scale = d3.scale.sqrt();
        this._scale.range([1, 30]);

        this._links = [];
        this._nodes = [];

        this._size = [960, 500];
        this._radius = 5;

        this._force = d3.layout.force()
            .nodes(this._nodes)
            .links(this._links)
            .charge(-120)
            .linkStrength(0.3)
            .linkDistance(function (d) { return self._scale(d.distance); })
            .size(this._size)
            .on('tick', this._forceTick.bind(this));
        this._resize();

        this._nodeSelect = this._svg.selectAll(".node");
        this._linkSelect = this._svg.selectAll(".link");

        var noresize = null;
        window.addEventListener('resize', function () {
            clearTimeout(noresize);
            noresize = setTimeout(self._resize.bind(self), 1000);
        });
    }
    inherits(GraphViewer, EventAbstraction);

    GraphViewer.prototype._resize = function () {
        var size = this._svg.node().getBoundingClientRect();
        var width = Math.floor(size.width),
            height = Math.floor(size.height);
        if (width !== this._size[0] || height !== this._size[1]) {
            this._size = [width, height];
            this._force.size(this._size);

            if (this._nodes.length > 0) {
                this._force.start();
            }
        }
    };

    GraphViewer.prototype._forceTick = function () {
        var self = this;
        var radius = this._radius;
        this._nodeSelect
            .attr("cx", function(d) {
                return Math.max(radius, Math.min(self._size[0] - radius, d.x));
            })
            .attr("cy", function(d) {
                return Math.max(radius, Math.min(self._size[1] - radius, d.y));
            });

        this._linkSelect.attr("x1", function(d) { return d.source.x; })
                        .attr("y1", function(d) { return d.source.y; })
                        .attr("x2", function(d) { return d.target.x; })
                        .attr("y2", function(d) { return d.target.y; });
    };

    GraphViewer.prototype._forceUpdate = function () {
        var self = this;

        this._linkSelect = this._linkSelect
            .data(this._force.links(), function(d) { return d.source.id + "-" + d.target.id; });
        this._linkSelect.enter().insert("line", ".node")
            .attr("class", "link");
        this._linkSelect.exit().remove();

        this._nodeSelect = this._nodeSelect
            .data(this._force.nodes(), function(d) { return d.id; });
        this._nodeSelect.enter().append("circle")
            .attr("class", "node")
            .attr("r", this._radius - 0.75)
            .style("fill", function(d) { return self._colors(d.website); })
            .call(this._force.drag)
            .call(this.enterNodeEventer.bind(this))
            .append("title")
                .text(function(d) { return d.title; });
        this._nodeSelect.exit()
            .call(this.exitNodeEventer.bind(this))
            .remove();

        this._force.start();
    };

    GraphViewer.prototype.update = function (data) {
        this._nodes.length = 0;
        this._links.length = 0;

        for (var g = 0; g < data.groups.length; g++) {
            var group = data.groups[g];

            for (var n = 0; n < group.nodes.length; n++) {
                this._nodes.push(group.nodes[n]);
            }
            for (var l = 0; l < group.links.length; l++) {
                this._links.push(group.links[l]);
            }
        }

        this._scale.domain([data.min, data.max]);
        this._forceUpdate();
    };

    //
    // Manage details view
    //
    function DetailsView(viewer) {
        if (!(this instanceof DetailsView)) return new DetailsView();

        this._sides = {
            left: {
                xhr: null, d: null, fetch: null,
                table: new PropertyTable()
            },
            right: {
                xhr: null, d: null, fetch: null,
                table: new PropertyTable()
            },
            center: {
                xhr: null, fetch: null,
                table: new PropertyTable()
            }
        };

        this._viewer = viewer;
        this._details = document.querySelector('#details');

        this._sides.left.table.append(this._details.querySelectorAll('.details-doc')[0]);
        this._sides.right.table.append(this._details.querySelectorAll('.details-doc')[1]);
        this._sides.center.table.append(this._details.querySelector('.details-center'));
    }

    DetailsView.prototype._update = function (d, side) {
        var self = this;

        var s = this._sides[side];
        var c = this._sides.center;

        if (s.xhr) s.xhr.abort();
        s.d = d;
        s.table.clear();
        s.xhr = d3.json('/article.json?id=' + d.id, function (error, result) {
            if (error) return self._viewer.status('error', 'Request error: ' + error.statusText);
            console.log(result);

            result.index = d.id;
            s.fetch = result;
            window.localStorage.setItem('graph-' + side, JSON.stringify(result));
            s.table.update({
                'index': d.id,
                'date': (new Date(result.date)).toISOString().slice(0, -5),
                'website': result.website
            });
        });

        if (this._sides.left.d && this._sides.right.d) {
            this._viewer.status('wait', 'Compareing nodes ...');

            if (c.xhr) c.xhr.abort();
            c.table.clear();
            c.xhr = d3.json('/compare.json?a=' + this._sides.left.d.id + '&b=' + this._sides.right.d.id, function (error, result) {
                if (error) return self._viewer.status('error', 'Request error: ' + error.statusText);
                self._viewer.status('node');

                c.fetch = result;
                c.table.update(result);
            });
        }
    };

    DetailsView.prototype.left = function (d) {
        this._update(d, 'left');
    };

    DetailsView.prototype.right = function (d) {
        this._update(d, 'right');
    };

    //
    // Property table
    //
    function PropertyTable() {
        this._table = document.createElement('table');
    }

    PropertyTable.prototype._createCell = function (val) {
        var td = document.createElement('td');
            td.appendChild(document.createTextNode(val));
        return td;
    };

    PropertyTable.prototype.append = function (el) {
        el.appendChild(this._table);
    };

    PropertyTable.prototype.clear = function () {
        this._table.innerHTML = '';
    };

    PropertyTable.prototype.update = function (content) {
        this.clear();

        var keys = Object.keys(content);
        for (var i = 0; i < keys.length; i++) {
            var tr = document.createElement('tr');
                tr.appendChild(this._createCell(keys[i]));
                tr.appendChild(this._createCell(content[keys[i]]));
            this._table.appendChild(tr);
        }
    };

    //
    // Event handler Abstraction
    //
    function EventAbstraction() {
        this._events = {
            nodeMap: {},
            listeners: {},
            keys: []
        };
    }

    EventAbstraction.prototype.addEventListener = function (eventName, fn) {
        this._events.listeners[eventName] = fn;
        this._events.keys.push(eventName);
    };

    EventAbstraction.prototype.enterNodeEventer = function (nodeCollection) {
        var self = this;

        for (var i = 0; i < this._events.keys.length; i++) {
            var eventName = this._events.keys[i];
            var listener = this._events.listeners[eventName];
            nodeCollection.on(eventName, listener);
        }

        nodeCollection.each(function (d) {
            self._events.nodeMap[d.id] = this;
        });
    };

    EventAbstraction.prototype.exitNodeEventer = function (nodeCollection) {
        var self = this;
        nodeCollection.each(function (d) {
            delete self._events.nodeMap[d.id];
        });
    };

    EventAbstraction.prototype.getNode = function (d) {
        return this._events.nodeMap[d.id];
    };

    //
    // Utils
    //
    function inherits(ctor, superCtor) {
        ctor.super_ = superCtor;
        ctor.prototype = Object.create(superCtor.prototype, {
            constructor: {
                value: ctor,
                enumerable: false,
                writable: true,
                configurable: true
            }
        });
    }

    function isElementVisible(el) {
        var rect     = el.getBoundingClientRect(),
            efp      = function (x, y) { return document.elementFromPoint(x, y); };

        // Return false if it's not in the viewport
        if (rect.right < 0 || rect.bottom < 0 || rect.left > window.innerWidth || rect.top > window.innerHeight) {
            return false;
        }

        var x = rect.left + rect.width/2,
            y = rect.top + rect.height/2;

        // Return true if any of its four corners are visible
        var eap = document.elementFromPoint(x, y);
        return (eap == el || el.contains(eap));
    }
})(window.d3, window.websites);
