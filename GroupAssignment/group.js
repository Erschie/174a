import {defs, tiny} from './examples/common.js';
import {Shape_From_File} from "./examples/obj-file-demo.js";
import {Color_Phong_Shader, Shadow_Textured_Phong_Shader,
    Depth_Texture_Shader_2D, Buffered_Texture, LIGHT_DEPTH_TEX_SIZE} from './examples/shadow-demo-shaders.js'

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene, Texture
} = tiny;

const {Cube, Axis_Arrows, Textured_Phong} = defs

export class Body {
    // **Body** can store and update the properties of a 3D body that incrementally
    // moves from its previous place due to velocities.  It conforms to the
    // approach outlined in the "Fix Your Timestep!" blog post by Glenn Fiedler.
    constructor(shape, material, size) {
        Object.assign(this,
            {shape, material, size})
    }

    // (within some margin of distance).
    static intersect_cube(p, margin = 0) {
        return p.every(value => value >= -1 - margin && value <= 1 + margin)
    }

    static intersect_sphere(p, margin = 0) {
        return p.dot(p) < 1 + margin;
    }

    emplace(location_matrix, linear_velocity, angular_velocity, spin_axis = vec3(0, 0, 0).randomized(1).normalized()) {                               // emplace(): assign the body's initial values, or overwrite them.
        this.center = location_matrix.times(vec4(0, 0, 0, 1)).to3();
        this.rotation = Mat4.translation(...this.center.times(-1)).times(location_matrix);
        this.previous = {center: this.center.copy(), rotation: this.rotation.copy()};
        // drawn_location gets replaced with an interpolated quantity:
        this.drawn_location = location_matrix;
        this.temp_matrix = Mat4.identity();
        return Object.assign(this, {linear_velocity, angular_velocity, spin_axis})
    }

    advance(time_amount) {
        // advance(): Perform an integration (the simplistic Forward Euler method) to
        // advance all the linear and angular velocities one time-step forward.
        this.previous = {center: this.center.copy(), rotation: this.rotation.copy()};
        // Apply the velocities scaled proportionally to real time (time_amount):
        // Linear velocity first, then angular:
        this.center = this.center.plus(this.linear_velocity.times(time_amount));
        this.rotation.pre_multiply(Mat4.rotation(time_amount * this.angular_velocity, ...this.spin_axis));
    }

    // The following are our various functions for testing a single point,
    // p, against some analytically-known geometric volume formula

    blend_rotation(alpha) {
        // blend_rotation(): Just naively do a linear blend of the rotations, which looks
        // ok sometimes but otherwise produces shear matrices, a wrong result.

        // TODO:  Replace this function with proper quaternion blending, and perhaps
        // store this.rotation in quaternion form instead for compactness.
        return this.rotation.map((x, i) => vec4(...this.previous.rotation[i]).mix(x, alpha));
    }

    blend_state(alpha) {
        // blend_state(): Compute the final matrix we'll draw using the previous two physical
        // locations the object occupied.  We'll interpolate between these two states as
        // described at the end of the "Fix Your Timestep!" blog post.
        this.drawn_location = Mat4.translation(...this.previous.center.mix(this.center, alpha))
            .times(this.blend_rotation(alpha))
            .times(Mat4.scale(...this.size));
    }

    check_if_colliding(b, collider) {
        // check_if_colliding(): Collision detection function.
        // DISCLAIMER:  The collision method shown below is not used by anyone; it's just very quick
        // to code.  Making every collision body an ellipsoid is kind of a hack, and looping
        // through a list of discrete sphere points to see if the ellipsoids intersect is *really* a
        // hack (there are perfectly good analytic expressions that can test if two ellipsoids
        // intersect without discretizing them into points).
        if (this == b)
            return false;
        // Nothing collides with itself.
        // Convert sphere b to the frame where a is a unit sphere:
        const T = this.inverse.times(b.drawn_location, this.temp_matrix);

        const {intersect_test, points, leeway} = collider;
        // For each vertex in that b, shift to the coordinate frame of
        // a_inv*b.  Check if in that coordinate frame it penetrates
        // the unit sphere at the origin.  Leave some leeway.
        return points.arrays.position.some(p =>
            intersect_test(T.times(p.to4(1)).to3(), leeway));
    }
}


export class Simulation extends Scene {
    // **Simulation** manages the stepping of simulation time.  Subclass it when making
    // a Scene that is a physics demo.  This technique is careful to totally decouple
    // the simulation from the frame rate (see below).
    constructor() {
        super();
        Object.assign(this, {time_accumulator: 0, time_scale: 1, t: 0, dt: 1 / 20, bodies: [], steps_taken: 0});
        this.light = 0;
    }

    simulate(frame_time) {
        // simulate(): Carefully advance time according to Glenn Fiedler's
        // "Fix Your Timestep" blog post.
        // This line gives ourselves a way to trick the simulator into thinking
        // that the display framerate is running fast or slow:
        frame_time = this.time_scale * frame_time;

        // Avoid the spiral of death; limit the amount of time we will spend
        // computing during this timestep if display lags:
        this.time_accumulator += Math.min(frame_time, 0.1);
        // Repeatedly step the simulation until we're caught up with this frame:
        while (Math.abs(this.time_accumulator) >= this.dt) {
            // Single step of the simulation for all bodies:
            this.update_state(this.dt);
            for (let b of this.bodies)
                b.advance(this.dt);
            // Following the advice of the article, de-couple
            // our simulation time from our frame rate:
            this.t += Math.sign(frame_time) * this.dt;
            this.time_accumulator -= Math.sign(frame_time) * this.dt;
            this.steps_taken++;
        }
        // Store an interpolation factor for how close our frame fell in between
        // the two latest simulation time steps, so we can correctly blend the
        // two latest states and display the result.
        let alpha = this.time_accumulator / this.dt;
        for (let b of this.bodies) b.blend_state(alpha);
    }

    make_control_panel() {
        // make_control_panel(): Create the buttons for interacting with simulation time.
        this.key_triggered_button("Speed up time", ["Shift", "T"], () => this.time_scale *= 5);
        this.key_triggered_button("Slow down time", ["t"], () => this.time_scale /= 5);
        this.new_line();
        this.live_string(box => {
            box.textContent = "Time scale: " + this.time_scale
        });
        this.new_line();
        this.live_string(box => {
            box.textContent = "Fixed simulation time step size: " + this.dt
        });
        this.new_line();
        this.live_string(box => {
            box.textContent = this.steps_taken + " timesteps were taken so far."
        });
        this.new_line();

    }

    texture_buffer_init(gl) {
        // Depth Texture
        this.lightDepthTexture = gl.createTexture();
        // Bind it to TinyGraphics
        this.light_depth_texture = new Buffered_Texture(this.lightDepthTexture);
        this.stars.light_depth_texture = this.light_depth_texture
        this.floor.light_depth_texture = this.light_depth_texture

        this.lightDepthTextureSize = LIGHT_DEPTH_TEX_SIZE;
        gl.bindTexture(gl.TEXTURE_2D, this.lightDepthTexture);
        gl.texImage2D(
            gl.TEXTURE_2D,      // target
            0,                  // mip level
            gl.DEPTH_COMPONENT, // internal format
            this.lightDepthTextureSize,   // width
            this.lightDepthTextureSize,   // height
            0,                  // border
            gl.DEPTH_COMPONENT, // format
            gl.UNSIGNED_INT,    // type
            null);              // data
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // Depth Texture Buffer
        this.lightDepthFramebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.lightDepthFramebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,       // target
            gl.DEPTH_ATTACHMENT,  // attachment point
            gl.TEXTURE_2D,        // texture target
            this.lightDepthTexture,         // texture
            0);                   // mip level
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // create a color texture of the same size as the depth texture
        // see article why this is needed_
        this.unusedTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.unusedTexture);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            this.lightDepthTextureSize,
            this.lightDepthTextureSize,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            null,
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // attach it to the framebuffer
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,        // target
            gl.COLOR_ATTACHMENT0,  // attachment point
            gl.TEXTURE_2D,         // texture target
            this.unusedTexture,         // texture
            0);                    // mip level
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }


    display(context, program_state) {
        // display(): advance the time and state of our whole simulation.
        if (program_state.animate)
            this.simulate(program_state.animation_delta_time);
        // Draw each shape at its current location:
        for (let b of this.bodies)
            b.shape.draw(context, program_state, b.drawn_location, b.material);
    }

    update_state(dt)      // update_state(): Your subclass of Simulation has to override this abstract function.
    {
        throw "Override this"
    }
}

export class Group extends Simulation {
    constructor() {
        super();

        this.meteorites = {
            rock: new defs.Subdivision_Sphere(1),
            sphere2: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(2),
            sphere3: new defs.Subdivision_Sphere(3),
            cube: new defs.Cube(),
        };

        this.shapes = {
            head: new defs.Subdivision_Sphere(5),
            rock: new defs.Subdivision_Sphere(1),
            cube: new defs.Cube(),
            sphere: new defs.Subdivision_Sphere(4),
            sphere2: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(2),
            sphere3: new defs.Subdivision_Sphere(3),
            moon: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(1),
            circle: new defs.Regular_2D_Polygon(1, 15),
            opm: new Shape_From_File("assets/saitama-ok-memechallenge/source/Saitama_OK_Cel_shaded/Saitama_OK_Cel_shaded.obj"),
            platform: new defs.Cube(),
            pillar: new defs.Cylindrical_Tube(10,10,[[0,1],[0,1]]),

        };

        this.materials = {
            rock: new Material(new defs.Phong_Shader(),
                {ambient: .4, specularity: 0.9, color: hex_color("#6c6c6c")}),
            skin: new Material(new defs.Phong_Shader(),
                {ambient: .15, specularity: 0.9, color: hex_color("#e0ac69")}),
            ring: new Material(new Ring_Shader(),
                {ambient: 1, diffusivity: 0, specularity: 0, color: hex_color("#b08040")}),
            body_suit: new Material(new defs.Phong_Shader(),
                {ambient: .15, specularity: 0.9, color: hex_color("#ffff00")}),
            eye: new Material(new defs.Phong_Shader(),
                {ambient: .15, specularity: 0.9, color: hex_color("#ffffff")}),
            opm: new Material(new Textured_Phong(),
                {   color: hex_color("#000000"),
                    ambient: 1.0, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/saitama-ok-memechallenge/textures/Saitama_OK_diffuse.png", "NEAREST")}),
            sun: new Material(new defs.Phong_Shader(),
                {ambient:1, color: hex_color("#ffffff")}),
            background_objects: new Material(new defs.Phong_Shader(),
                {ambient:0.2, diffusivity: 0.5, specularity:0.5, color: hex_color("#ffffff")}),

        }
        this.colliders = [
            {intersect_test: Body.intersect_sphere, points: new defs.Subdivision_Sphere(1), leeway: 1},
            {intersect_test: Body.intersect_sphere, points: new defs.Subdivision_Sphere(2), leeway: .3},
            {intersect_test: Body.intersect_cube, points: new defs.Cube(), leeway: .1}
        ];
        this.collider_selection = 0;
        let opm_scale = Mat4.scale(7,7,7);
        let opm_rot = Mat4.rotation(0.5, 0,1,0);
        this.opm = new Body(this.shapes.opm, this.materials.opm, vec3(5,5,5))
            .emplace(opm_scale.times(opm_rot).times(Mat4.identity()),
                0, 0);
        this.initial_camera_location = Mat4.look_at(vec3(0, 0, 20), vec3(0, 0, 0), vec3(0, 1, 0));
        this.r = 0.917;
        this.g = 0.792;
        this.b = 0.949;
    }

    random_shape(shape_list = this.meteorites) {
        // random_shape():  Extract a random shape from this.shapes.
        const shape_names = Object.keys(shape_list);
        return shape_list[shape_names[~~(shape_names.length * Math.random())]]
    }

    increase() {
        this.collider_selection = Math.min(this.collider_selection + 1, this.colliders.length - 1);
    }

    decrease() {
        this.collider_selection = Math.max(this.collider_selection - 1, 0)
    }

    update_state(dt) {
        // update_state():  Override the base time-stepping code to say what this particular
        // scene should do to its bodies every frame -- including applying forces.
        // Generate additional moving bodies if there ever aren't enough:
        while (this.bodies.length < 25)
            this.bodies.push(new Body(this.random_shape(), this.materials.rock, vec3(1, 1 + Math.random(), 1))
                .emplace(Mat4.translation(...vec3(50, 15, 0).randomized(10)),
                    vec3(0, -1, 0).randomized(2).normalized().times(2), Math.random()));

        const collider = this.colliders[this.collider_selection];

        for (let a of this.bodies) {
            // Cache the inverse of matrix of body "a" to save time.
            a.inverse = Mat4.inverse(a.drawn_location);

            //a.linear_velocity = a.linear_velocity.minus(a.center.times(dt));
            a.linear_velocity[0] += dt * -2;
            if (a.linear_velocity.norm() == 0)
                continue;
            // *** Collision process is here ***
            // Loop through all bodies again (call each "b"):

            // Pass the two bodies and the collision shape to check_if_colliding():
            if (!a.check_if_colliding(this.opm, collider))
                continue;
            // If we get here, we collided, so turn red and zero out the
            // velocity so they don't inter-penetrate any further.
            //a.material = this.active_color;
            a.linear_velocity[0] *= -.8;
            //a.angular_velocity = 0;
        }

        /*for (let b of this.bodies) {
            // Gravity on Earth, where 1 unit in world space = 1 meter:
            b.linear_velocity[0] += dt * -9.8;
            // If about to fall through floor, reverse y velocity:
            if (b.center[0] < 1 && b.linear_velocity[1] < 0)
                b.linear_velocity[0] *= -.8;
        }*/
        // Delete bodies that stop or stray too far away:
        //this.bodies = this.bodies.filter(b => b.center.norm() < 50 && b.linear_velocity.norm() > 2);
        this.bodies = this.bodies.filter(b => b.center.norm() < 50 && b.linear_velocity.norm() > 2);
    }

    make_control_panel() {
        this.key_triggered_button("Change Light Color", ["l"], () =>{
            this.r = Math.random();
            this.g = Math.random();
            this.b = Math.random();
        });
        this.new_line();
        this.key_triggered_button("Original Light Color", ['o'], () => {
            this.r = 0.917;
            this.g = 0.792;
            this.b = 0.949;
        })
    }

    display(context, program_state) {
        super.display(context, program_state);
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            program_state.set_camera(Mat4.translation(0, 0, -50));
        }

        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);

        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;
        const ts = program_state.animation_time / 100;
        const tss = program_state.animation_time / 10;

        // The parameters of the Light are: position, color, size
        const light_position = vec4(36, 21, 0, 1);

        //const light_radius = 5;
        const sun_radius = 5;
        const light_size = 10**sun_radius;

        // TODO: Add the background environment
        // TODO: shadows?

        // Added the ability to change the sun's color to two different colors: yellow and purple

        //this is the color purple
        let sun_color = color(this.r, this.g, this.b, 1);
        if (Math.floor(ts) % 2 === 0) {
            //sun_color = color(0.882, 0.666, 0.933,1);
            sun_color = color(this.r+0.1, this.g+0.1, this.b+0.1, 1);
        }

        //this is the color yellow
        /*if (this.light == 1)
        {
            sun_color = color(Math.random(), Math.random(), Math.random())
            //sun_color = color(1, 1, 0.1, 1);
            if (Math.floor(ts) % 2 === 0)
            {
                sun_color = sun_color + vec3(0.05, 0.05, 0.05);
            }
            else {
                sun_color = sun_color - vec3(0.05, 0.05, 0.05);
            }
        }*/

        //program_state.projection_transform = Mat4.perspective(Math.PI / 4, context.width / context.height, 1, 500);

        program_state.lights = [new Light(light_position, sun_color, light_size)];

        let model_transform = Mat4.identity();
        let sun_tr = Mat4.translation(36,21,0);
        let sun_sc = Mat4.scale(sun_radius, sun_radius, sun_radius);
        let sun_transform = sun_tr.times(sun_sc).times(model_transform);
        this.opm.shape.draw(context, program_state, this.opm.drawn_location, this.materials.opm);
        this.shapes.sphere.draw(context, program_state, sun_transform, this.materials.sun.override({color: sun_color}));

        // This section adds the background environment
        // The background consists of a cylinder and a floor
        // That is literally it.

        //This part adds a floor
        let model_transform_floor = model_transform.times(Mat4.scale(40, 1,30)).times(Mat4.translation(0,-18,0));
        this.shapes.platform.draw(context, program_state, model_transform_floor, this.materials.background_objects);

        //This part adds the cylinder background
        let model_transform_cylinder = model_transform.times(Mat4.translation(-21,10,-20)).times(Mat4.scale(10, 60, 10)).times(Mat4.rotation(55, 1,0,0));
        this.shapes.pillar.draw(context, program_state, model_transform_cylinder, this.materials.background_objects);

        // This section adds in the shadows


        // set up camera
        if (this.attached) {
            if (this.attached() == this.initial_camera_location) {
                let desired = this.initial_camera_location;
                desired = desired.map((x, i) => Vector.from(program_state.camera_inverse[i]).mix(x, 0.1))
                program_state.set_camera(desired);
            } else {
                let desired = Mat4.inverse(this.attached().times(Mat4.translation(0, 0, 5)));
                desired = desired.map((x, i) => Vector.from(program_state.camera_inverse[i]).mix(x, 0.1))
                program_state.set_camera(desired);
            }
        }
    }
}

class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;

        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, vertex_worldspace;
        
        // for gourand shader
        varying vec4 vertex_color;
        
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                
                // for gouraud shader
                vertex_color = vec4( shape_color.xyz * ambient, shape_color.w );
                vertex_color.xyz = phong_model_lights( normalize( N ), vertex_worldspace );
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){                                                           
                // // Compute an initial (ambient) color:
                // gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                // // Compute the final color with contributions from lights:
                // gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                
                // for gouraud shader
                gl_FragColor = vertex_color;
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

class Ring_Shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 point_position;
        varying vec4 center;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        
        void main(){
            // The vertex's final resting place (in NDCS):
            gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
            point_position = model_transform * vec4( position, 1.0 );
            
            // center of ring
            center = model_transform * vec4( 0.0, 0.0, 0.0, 1.0 );
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){
            vec3 distance = vec3( point_position.xyz - center.xyz );
            
            gl_FragColor = vec4( vec3(0.816, 0.447, 0.318), cos( length(distance) * 20.0 ) );
        }`;
    }
}

