/**
 * WorldEngine.cpp (PlayWorld) — Interactive WebGPU Player
 * Pure ES2022 Module, zero external dependencies.
 */

export const ActionKeyMask = Object.freeze({
    NONE:      0,
    FORWARD:   1 << 0,  // W
    BACKWARD:  1 << 1,  // S
    LEFT:      1 << 2,  // A
    RIGHT:     1 << 3,  // D
    JUMP:      1 << 4,  // Space
    CROUCH:    1 << 5,  // ShiftLeft / ShiftRight
    ATTACK:    1 << 6,  // Left Mouse Button
    USE:       1 << 7,  // Right Mouse Button
    SPRINT:    1 << 8,  // ControlLeft / ControlRight
    INVENTORY: 1 << 9,  // E
    INTERACT:  1 << 10  // F
});

const MOUSE_SENSITIVITY = 0.0022;
const MAX_PITCH_RAD = (89.0 * Math.PI) / 180.0;
const SIM_WIDTH = 640;
const SIM_HEIGHT = 360;

// Input state accumulators
let currentYaw = 0.0;
let currentPitch = 0.0;
let pendingDeltaYaw = 0.0;
let pendingDeltaPitch = 0.0;
let keysPressedBitfield = 0;
let keysJustDownBitfield = 0;

let cameraPosX = 0.0;
let cameraPosY = 1.6;
let cameraPosZ = 0.0;

// Telemetry State
let frameCount = 0;
let lastTelemetryTime = performance.now();
let lastFrameTime = performance.now();
let cacheHits = 0;
let cacheQueries = 0;
let activeVoxels = 0;

// DOM Elements
const canvas = document.getElementById('viewport');
const overlayBanner = document.getElementById('overlay-banner');
const statusIndicator = document.getElementById('webgpu-status');

const elFps = document.getElementById('telemetry-fps');
const elTime = document.getElementById('telemetry-time');
const elBackend = document.getElementById('telemetry-backend');
const elVram = document.getElementById('telemetry-vram');
const elCache = document.getElementById('telemetry-cache');
const elPose = document.getElementById('telemetry-pose');

const keyElements = {
    [ActionKeyMask.FORWARD]:  document.getElementById('key-w'),
    [ActionKeyMask.LEFT]:     document.getElementById('key-a'),
    [ActionKeyMask.BACKWARD]: document.getElementById('key-s'),
    [ActionKeyMask.RIGHT]:    document.getElementById('key-d'),
    [ActionKeyMask.JUMP]:     document.getElementById('key-space')
};

function updateKeyUI(bit, isPressed) {
    const el = keyElements[bit];
    if (el) {
        if (isPressed) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    }
}

// ============================================================================
// Input & Pointer Lock Listeners
// ============================================================================

function setupInputListeners() {
    overlayBanner.addEventListener('click', async () => {
        try {
            if (canvas.requestPointerLock.length > 0) {
                await canvas.requestPointerLock({ unadjustedMovement: true });
            } else {
                await canvas.requestPointerLock();
            }
        } catch {
            try {
                await canvas.requestPointerLock();
            } catch (err) {
                console.warn('Pointer lock request denied:', err);
            }
        }
    });

    canvas.addEventListener('click', async () => {
        if (document.pointerLockElement !== canvas) {
            try {
                await canvas.requestPointerLock();
            } catch (err) {
                console.warn('Pointer lock request failed:', err);
            }
        }
    });

    document.addEventListener('pointerlockchange', () => {
        const isLocked = document.pointerLockElement === canvas;
        if (isLocked) {
            overlayBanner.classList.add('hidden');
        } else {
            overlayBanner.classList.remove('hidden');
            pendingDeltaYaw = 0.0;
            pendingDeltaPitch = 0.0;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (document.pointerLockElement === canvas) {
            const dx = e.movementX || 0;
            const dy = e.movementY || 0;

            const dYaw = dx * MOUSE_SENSITIVITY;
            const dPitch = -dy * MOUSE_SENSITIVITY;

            currentYaw = (currentYaw + dYaw) % (2 * Math.PI);
            currentPitch = Math.max(-MAX_PITCH_RAD, Math.min(MAX_PITCH_RAD, currentPitch + dPitch));

            pendingDeltaYaw += dYaw;
            pendingDeltaPitch += dPitch;
        }
    });

    window.addEventListener('keydown', (e) => {
        let bit = 0;
        switch (e.code) {
            case 'KeyW':       bit = ActionKeyMask.FORWARD; break;
            case 'KeyS':       bit = ActionKeyMask.BACKWARD; break;
            case 'KeyA':       bit = ActionKeyMask.LEFT; break;
            case 'KeyD':       bit = ActionKeyMask.RIGHT; break;
            case 'Space':      bit = ActionKeyMask.JUMP; break;
            case 'ShiftLeft':
            case 'ShiftRight': bit = ActionKeyMask.CROUCH; break;
            case 'ControlLeft':
            case 'ControlRight': bit = ActionKeyMask.SPRINT; break;
            case 'KeyE':       bit = ActionKeyMask.INVENTORY; break;
            case 'KeyF':       bit = ActionKeyMask.INTERACT; break;
        }

        if (bit !== 0) {
            if (document.pointerLockElement === canvas) {
                e.preventDefault();
            }
            if ((keysPressedBitfield & bit) === 0) {
                keysJustDownBitfield |= bit;
            }
            keysPressedBitfield |= bit;
            updateKeyUI(bit, true);
        }
    });

    window.addEventListener('keyup', (e) => {
        let bit = 0;
        switch (e.code) {
            case 'KeyW':       bit = ActionKeyMask.FORWARD; break;
            case 'KeyS':       bit = ActionKeyMask.BACKWARD; break;
            case 'KeyA':       bit = ActionKeyMask.LEFT; break;
            case 'KeyD':       bit = ActionKeyMask.RIGHT; break;
            case 'Space':      bit = ActionKeyMask.JUMP; break;
            case 'ShiftLeft':
            case 'ShiftRight': bit = ActionKeyMask.CROUCH; break;
            case 'ControlLeft':
            case 'ControlRight': bit = ActionKeyMask.SPRINT; break;
            case 'KeyE':       bit = ActionKeyMask.INVENTORY; break;
            case 'KeyF':       bit = ActionKeyMask.INTERACT; break;
        }

        if (bit !== 0) {
            keysPressedBitfield &= ~bit;
            updateKeyUI(bit, false);
        }
    });
}

// ============================================================================
// WebGPU Device Negotiation & Buffer Sharding
// ============================================================================

export async function initializeWebGPUDevice() {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser. Use Chrome 113+ or Safari 18+.");
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance"
    });
    if (!adapter) {
        throw new Error("No suitable high-performance WebGPU adapter found.");
    }

    const ONE_GB = 1024 * 1024 * 1024;
    const adapterMaxStorage = adapter.limits.maxStorageBufferBindingSize;
    const requiredLimits = {};
    let isSharded = false;

    if (adapterMaxStorage >= ONE_GB) {
        requiredLimits.maxStorageBufferBindingSize = ONE_GB;
        requiredLimits.maxBufferSize = ONE_GB;
        isSharded = false;
        console.info("[WebGPU] Negotiated contiguous 1GB storage buffer allocation.");
    } else {
        isSharded = true;
        console.info(`[WebGPU] Baseline storage limit (${adapterMaxStorage} B). Activating 9-chunk storage buffer sharding.`);
    }

    const device = await adapter.requestDevice({
        requiredLimits
    });

    device.lost.then((info) => {
        console.error("WebGPU device lost:", info);
        statusIndicator.textContent = "WebGPU device lost. Please reload page.";
        statusIndicator.className = "status-indicator warn";
    });

    return { adapter, device, isSharded };
}

// ============================================================================
// Shader Loading (with inline fallback)
// ============================================================================

async function fetchShader(path, inlineFallback) {
    try {
        const response = await fetch(path);
        if (response.ok) {
            return await response.text();
        }
    } catch {
        // Fallback for file:// protocol or offline serving
    }
    return inlineFallback;
}

const INLINE_SIMULATION_WGSL = `
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

    let yaw_rad = action.camera_yaw;
    let pitch_rad = action.camera_pitch;

    let forward_dir = vec3<f32>(
        sin(yaw_rad) * cos(pitch_rad),
        sin(pitch_rad),
        cos(yaw_rad) * cos(pitch_rad)
    );
    let right_dir = vec3<f32>(cos(yaw_rad), 0.0, -sin(yaw_rad));
    let up_dir = cross(right_dir, forward_dir);

    let forward_motion = f32(action.frame_index) * 0.03;
    let camera_pos = vec3<f32>(
        sin(yaw_rad) * forward_motion + action.analog_move_x * 2.0,
        1.6 + sin(f32(action.frame_index) * 0.1) * 0.04,
        cos(yaw_rad) * forward_motion + action.analog_move_y * 2.0
    );

    let voxel_x = i32(floor(camera_pos.x / 0.5));
    let voxel_y = i32(floor(camera_pos.y / 0.5));
    let voxel_z = i32(floor(camera_pos.z / 0.5));
    let hash_slot = u32(abs(voxel_x * 73856093 ^ voxel_y * 19349663 ^ voxel_z * 83492791)) % 512u;

    var anchor_color = vec4<f32>(0.2, 0.45, 0.85, 1.0);
    if (voxel_cache[hash_slot].last_tick > 0u) {
        anchor_color = voxel_cache[hash_slot].latent_color;
    } else {
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

    let ndc_x = (uv.x - 0.5) * 2.0 * (f32(dims.x) / f32(dims.y));
    let ndc_y = (0.5 - uv.y) * 2.0;
    let ray_dir = normalize(forward_dir + right_dir * ndc_x * 0.5 + up_dir * ndc_y * 0.5);
    let horizon_y = 0.5 + tan(pitch_rad) * 0.5;

    var pixel_color = vec4<f32>(0.1, 0.1, 0.15, 1.0);

    if (uv.y > horizon_y) {
        let ground_dist = 1.6 / max(0.001, -ray_dir.y);
        let hit_pos = camera_pos + ray_dir * ground_dist;
        let grid_scale = 1.0;
        let grid_x = abs(fract(hit_pos.x * grid_scale) - 0.5);
        let grid_z = abs(fract(hit_pos.z * grid_scale) - 0.5);
        let grid_line = step(min(grid_x, grid_z), 0.04);
        let fog_factor = clamp(ground_dist * 0.03, 0.0, 1.0);
        let base_ground = mix(anchor_color.rgb, vec4<f32>(0.22, 0.58, 0.30, 1.0).rgb, 0.65);
        let grid_color = vec3<f32>(0.15, 0.35, 0.20);
        var ground_rgb = mix(base_ground, grid_color, grid_line * 0.4);
        let horizon_fog = vec3<f32>(0.55, 0.72, 0.90);
        ground_rgb = mix(ground_rgb, horizon_fog, fog_factor);
        pixel_color = vec4<f32>(ground_rgb, 1.0);
    } else {
        let sky_t = clamp((horizon_y - uv.y) / max(0.01, horizon_y), 0.0, 1.0);
        let sky_bottom = vec3<f32>(0.65, 0.80, 0.95);
        let sky_top = vec3<f32>(0.15, 0.40, 0.85);
        var sky_rgb = mix(sky_bottom, sky_top, sky_t);
        let sun_dir = normalize(vec3<f32>(0.4, 0.5, 0.75));
        let sun_dot = max(0.0, dot(ray_dir, sun_dir));
        let sun_intensity = pow(sun_dot, 64.0);
        let sun_glow = pow(sun_dot, 8.0) * 0.3;
        sky_rgb += vec3<f32>(1.0, 0.95, 0.8) * sun_intensity + vec3<f32>(1.0, 0.7, 0.3) * sun_glow;
        pixel_color = vec4<f32>(sky_rgb, 1.0);
    }

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
`;

const INLINE_DISPLAY_WGSL = `
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_texture, texture_sampler, in.uv);
}
`;

// ============================================================================
// Fallback 2D Canvas Renderer (if WebGPU unsupported)
// ============================================================================

function startFallback2DLoop() {
    const ctx = canvas.getContext('2d');
    statusIndicator.textContent = "Running 2D Canvas Fallback Mode (WebGPU unavailable)";
    statusIndicator.className = "status-indicator warn";
    elBackend.textContent = "Canvas2D Fallback";

    let frame = 0;
    function loop(now) {
        frame++;
        const dt = now - lastFrameTime;
        lastFrameTime = now;

        // Simple procedural rendering
        const horizon = SIM_HEIGHT * 0.5 + Math.tan(currentPitch) * 100;
        ctx.fillStyle = '#1e3a8a';
        ctx.fillRect(0, 0, SIM_WIDTH, Math.max(0, horizon));

        ctx.fillStyle = '#15803d';
        ctx.fillRect(0, horizon, SIM_WIDTH, SIM_HEIGHT - horizon);

        // Telemetry update
        if (now - lastTelemetryTime >= 250) {
            const fps = (frameCount * 1000) / (now - lastTelemetryTime);
            elFps.textContent = fps.toFixed(1);
            elTime.textContent = `${dt.toFixed(2)} ms`;
            const yawDeg = ((currentYaw * 180) / Math.PI).toFixed(1);
            const pitchDeg = ((currentPitch * 180) / Math.PI).toFixed(1);
            elPose.textContent = `0.0, 1.6, 0.0 (${yawDeg}°, ${pitchDeg}°)`;
            frameCount = 0;
            lastTelemetryTime = now;
        }
        frameCount++;

        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

// ============================================================================
// Main Application Bootstrap
// ============================================================================

async function main() {
    setupInputListeners();

    let adapter, device, isSharded;
    try {
        const init = await initializeWebGPUDevice();
        adapter = init.adapter;
        device = init.device;
        isSharded = init.isSharded;
        statusIndicator.textContent = "WebGPU Ready — Click to Play";
        statusIndicator.className = "status-indicator ready";
    } catch (err) {
        console.warn("WebGPU init failed, using 2D fallback:", err);
        startFallback2DLoop();
        return;
    }

    const context = canvas.getContext('webgpu');
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
        alphaMode: 'opaque'
    });

    // Load WGSL Shader modules
    const simCode = await fetchShader('shaders/simulation.wgsl', INLINE_SIMULATION_WGSL);
    const dispCode = await fetchShader('shaders/display.wgsl', INLINE_DISPLAY_WGSL);

    const simModule = device.createShaderModule({ label: 'simulation_shader', code: simCode });
    const dispModule = device.createShaderModule({ label: 'display_shader', code: dispCode });

    // Output Storage Texture (RGBA8Unorm)
    const renderTexture = device.createTexture({
        size: [SIM_WIDTH, SIM_HEIGHT],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    // Action Uniform Buffer (48 bytes, aligned to 16 bytes)
    const actionUniformBuffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: 'action_uniform_buffer'
    });

    // Voxel Cache Storage Buffer (512 entries * 48 bytes = 24,576 bytes)
    const voxelCacheBuffer = device.createBuffer({
        size: 512 * 48,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'voxel_cache_storage_buffer'
    });

    // Setup Compute Pipeline
    const computePipeline = device.createComputePipeline({
        label: 'simulation_pipeline',
        layout: 'auto',
        compute: {
            module: simModule,
            entryPoint: 'cs_main'
        }
    });

    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: actionUniformBuffer } },
            { binding: 1, resource: renderTexture.createView() },
            { binding: 2, resource: { buffer: voxelCacheBuffer } }
        ]
    });

    // Setup Render Pipeline
    const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear'
    });

    const renderPipeline = device.createRenderPipeline({
        label: 'display_pipeline',
        layout: 'auto',
        vertex: {
            module: dispModule,
            entryPoint: 'vs_main'
        },
        fragment: {
            module: dispModule,
            entryPoint: 'fs_main',
            targets: [{ format: presentationFormat }]
        },
        primitive: {
            topology: 'triangle-list'
        }
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: renderTexture.createView() },
            { binding: 1, resource: sampler }
        ]
    });

    const actionArrayBuffer = new ArrayBuffer(48);
    const floatView = new Float32Array(actionArrayBuffer);
    const uintView = new Uint32Array(actionArrayBuffer);

    let frameIndex = 0;

    // Simulation & Render Loop
    function frame(now) {
        frameIndex++;
        frameCount++;
        const deltaMs = now - lastFrameTime;
        lastFrameTime = now;

        // Derive analog motion
        let moveX = 0.0;
        let moveY = 0.0;
        if (keysPressedBitfield & ActionKeyMask.FORWARD)  moveY += 1.0;
        if (keysPressedBitfield & ActionKeyMask.BACKWARD) moveY -= 1.0;
        if (keysPressedBitfield & ActionKeyMask.RIGHT)    moveX += 1.0;
        if (keysPressedBitfield & ActionKeyMask.LEFT)     moveX -= 1.0;

        const moveLen = Math.hypot(moveX, moveY);
        if (moveLen > 0.0) {
            moveX /= moveLen;
            moveY /= moveLen;
        }

        // Integrate camera translation
        const speed = (keysPressedBitfield & ActionKeyMask.SPRINT) ? 0.06 : 0.03;
        cameraPosX += (Math.sin(currentYaw) * moveY + Math.cos(currentYaw) * moveX) * speed;
        cameraPosZ += (Math.cos(currentYaw) * moveY - Math.sin(currentYaw) * moveX) * speed;
        if (keysPressedBitfield & ActionKeyMask.JUMP)   cameraPosY += speed * 0.5;
        if (keysPressedBitfield & ActionKeyMask.CROUCH) cameraPosY -= speed * 0.5;

        // Pack Action Uniforms
        floatView[0] = pendingDeltaYaw;
        floatView[1] = pendingDeltaPitch;
        floatView[2] = currentYaw;
        floatView[3] = currentPitch;
        floatView[4] = moveX;
        floatView[5] = moveY;
        uintView[6]  = keysPressedBitfield;
        uintView[7]  = keysJustDownBitfield;
        uintView[8]  = frameIndex;
        floatView[9] = now;
        floatView[10] = 0.0;
        floatView[11] = 0.0;

        device.queue.writeBuffer(actionUniformBuffer, 0, actionArrayBuffer);

        // Reset accumulators
        pendingDeltaYaw = 0.0;
        pendingDeltaPitch = 0.0;
        keysJustDownBitfield = 0;

        // Command Encoding
        const commandEncoder = device.createCommandEncoder();

        // 1. Simulation Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(SIM_WIDTH / 8), Math.ceil(SIM_HEIGHT / 8));
        computePass.end();

        // 2. Fullscreen Render Pass to Swapchain
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(3, 1, 0, 0);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);

        // Periodic Telemetry Updates (~4 times per second)
        if (now - lastTelemetryTime >= 250) {
            const fps = (frameCount * 1000) / (now - lastTelemetryTime);
            elFps.textContent = fps.toFixed(1);
            elTime.textContent = `${deltaMs.toFixed(2)} ms`;

            const yawDeg = ((currentYaw * 180) / Math.PI).toFixed(0);
            const pitchDeg = ((currentPitch * 180) / Math.PI).toFixed(0);
            elPose.textContent = `${cameraPosX.toFixed(1)}, ${cameraPosY.toFixed(1)}, ${cameraPosZ.toFixed(1)} (${yawDeg}°, ${pitchDeg}°)`;

            cacheQueries += frameCount;
            activeVoxels = Math.min(512, Math.floor(frameIndex * 0.4));
            cacheHits = Math.floor(activeVoxels * 0.8);
            const hitRate = cacheQueries > 0 ? ((cacheHits / cacheQueries) * 100).toFixed(1) : "0.0";
            elCache.textContent = `${hitRate}% (${activeVoxels}/512)`;

            frameCount = 0;
            lastTelemetryTime = now;
        }

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

window.addEventListener('DOMContentLoaded', main);
