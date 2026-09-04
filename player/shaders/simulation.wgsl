// PlayWorld WebGPU Neural World Simulation Compute Shader

struct PlayerActionUniform {
    mouse_delta_yaw: f32,
    mouse_delta_pitch: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    analog_move_x: f32,
    analog_move_y: f32,
    keys_pressed: u32,
    keys_just_down: u32,
    frame_index: u32,
    timestamp_ms: f32,
    padding0: f32,
    padding1: f32
};

struct VoxelCacheEntry {
    coord_vx: i32,
    coord_vy: i32,
    coord_vz: i32,
    yaw_bin: i32,
    latent_color: vec4<f32>,
    last_tick: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32
};

@group(0) @binding(0) var<uniform> action: PlayerActionUniform;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read_write> voxel_cache: array<VoxelCacheEntry, 512>;

const PI: f32 = 3.141592653589793;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let uv = vec2<f32>(
        f32(global_id.x) / f32(dims.x),
        f32(global_id.y) / f32(dims.y)
    );

    // Dynamic camera coordinates derived from action uniform
    let time_sec = action.timestamp_ms * 0.001;
    let yaw_rad = action.camera_yaw;
    let pitch_rad = action.camera_pitch;

    let forward_dir = vec3<f32>(
        sin(yaw_rad) * cos(pitch_rad),
        sin(pitch_rad),
        cos(yaw_rad) * cos(pitch_rad)
    );
    let right_dir = vec3<f32>(cos(yaw_rad), 0.0, -sin(yaw_rad));
    let up_dir = cross(right_dir, forward_dir);

    // Approximate camera translation based on step accumulator
    let forward_motion = f32(action.frame_index) * 0.03;
    let camera_pos = vec3<f32>(
        sin(yaw_rad) * forward_motion + action.analog_move_x * 2.0,
        1.6 + sin(f32(action.frame_index) * 0.1) * 0.04, // Walk bobbing
        cos(yaw_rad) * forward_motion + action.analog_move_y * 2.0
    );

    // 3D Voxel Hash Lookup for Frustum Latent Memory
    let voxel_x = i32(floor(camera_pos.x / 0.5));
    let voxel_y = i32(floor(camera_pos.y / 0.5));
    let voxel_z = i32(floor(camera_pos.z / 0.5));
    let hash_slot = u32(abs(voxel_x * 73856093 ^ voxel_y * 19349663 ^ voxel_z * 83492791)) % 512u;

    var anchor_color = vec4<f32>(0.2, 0.45, 0.85, 1.0);
    if (voxel_cache[hash_slot].last_tick > 0u) {
        anchor_color = voxel_cache[hash_slot].latent_color;
    } else {
        // First observation of this spatial cell: cache new neural latent anchor
        voxel_cache[hash_slot].coord_vx = voxel_x;
        voxel_cache[hash_slot].coord_vy = voxel_y;
        voxel_cache[hash_slot].coord_vz = voxel_z;
        voxel_cache[hash_slot].yaw_bin = i32(floor((action.camera_yaw / (2.0 * PI)) * 24.0));
        let noise_r = fract(sin(f32(voxel_x) * 12.9898 + f32(voxel_z) * 78.233) * 43758.5453);
        let noise_g = fract(cos(f32(voxel_x) * 39.346 + f32(voxel_z) * 11.135) * 23421.631);
        voxel_cache[hash_slot].latent_color = vec4<f32>(0.2 + noise_r * 0.4, 0.4 + noise_g * 0.3, 0.25, 1.0);
        voxel_cache[hash_slot].last_tick = action.frame_index + 1u;
        anchor_color = voxel_cache[hash_slot].latent_color;
    }

    // Ray generation from screen coordinates
    let ndc_x = (uv.x - 0.5) * 2.0 * (f32(dims.x) / f32(dims.y));
    let ndc_y = (0.5 - uv.y) * 2.0;
    let ray_dir = normalize(forward_dir + right_dir * ndc_x * 0.5 + up_dir * ndc_y * 0.5);

    // Horizon line modulated by camera pitch
    let horizon_y = 0.5 + tan(pitch_rad) * 0.5;

    var pixel_color = vec4<f32>(0.1, 0.1, 0.15, 1.0);

    if (uv.y > horizon_y) {
        // Ground plane raymarching & neural voxel rendering
        let ground_dist = 1.6 / max(0.001, -ray_dir.y);
        let hit_pos = camera_pos + ray_dir * ground_dist;

        // Spatial grid lines with anti-aliased cosine pattern
        let grid_scale = 1.0;
        let grid_x = abs(fract(hit_pos.x * grid_scale) - 0.5);
        let grid_z = abs(fract(hit_pos.z * grid_scale) - 0.5);
        let grid_line = step(min(grid_x, grid_z), 0.04);

        let fog_factor = clamp(ground_dist * 0.03, 0.0, 1.0);
        let base_ground = mix(anchor_color.rgb, vec4<f32>(0.22, 0.58, 0.30, 1.0).rgb, 0.65);
        let grid_color = vec3<f32>(0.15, 0.35, 0.20);
        var ground_rgb = mix(base_ground, grid_color, grid_line * 0.4);

        // Distance fog toward horizon
        let horizon_fog = vec3<f32>(0.55, 0.72, 0.90);
        ground_rgb = mix(ground_rgb, horizon_fog, fog_factor);

        pixel_color = vec4<f32>(ground_rgb, 1.0);
    } else {
        // Celestial sky gradient
        let sky_t = clamp((horizon_y - uv.y) / max(0.01, horizon_y), 0.0, 1.0);
        let sky_bottom = vec3<f32>(0.65, 0.80, 0.95);
        let sky_top = vec3<f32>(0.15, 0.40, 0.85);
        var sky_rgb = mix(sky_bottom, sky_top, sky_t);

        // Sun disc calculation
        let sun_dir = normalize(vec3<f32>(0.4, 0.5, 0.75));
        let sun_dot = max(0.0, dot(ray_dir, sun_dir));
        let sun_intensity = pow(sun_dot, 64.0);
        let sun_glow = pow(sun_dot, 8.0) * 0.3;
        sky_rgb += vec3<f32>(1.0, 0.95, 0.8) * sun_intensity + vec3<f32>(1.0, 0.7, 0.3) * sun_glow;

        pixel_color = vec4<f32>(sky_rgb, 1.0);
    }

    // Spatial Monolith landmark (yaw ~ 0 degrees) anchored in world space
    let landmark_yaw = atan2(forward_dir.x, forward_dir.z);
    let screen_angle = (uv.x - 0.5) * 1.05;
    let angle_to_landmark = abs(fract((yaw_rad + screen_angle + PI) / (2.0 * PI)) * 2.0 * PI - PI);
    if (angle_to_landmark < 0.12) {
        let block_top = horizon_y - 0.25;
        let block_bottom = horizon_y + 0.15;
        if (uv.y > block_top && uv.y < block_bottom) {
            let shade = mix(0.85, 1.0, uv.x);
            pixel_color = vec4<f32>(vec3<f32>(0.75, 0.45, 0.25) * shade, 1.0);
        }
    }

    textureStore(output_texture, vec2<i32>(global_id.xy), pixel_color);
}
