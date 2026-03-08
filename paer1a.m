
%% Part (a) 
clear; close all; clc;

%% -------------------- 0) Parameters --------------------
v   = 343;            % speed of sound (m/s)
fs  = 16000;          % Part I sampling rate
dur = 4;              % seconds
Ns  = dur*fs;
t   = (0:Ns-1)'/fs;

L   = 2;              % ellipse semi-axis (m)
w0  = pi/2;           % angular speed (rad/s)
d   = 0.18;           % mic spacing (m)
r0  = 1;              % reference distance (m)

mic1 = [-d/2, 0];
mic2 = [ d/2, 0];

% Ellipse: x = 2L cos(wt), y = L sin(wt)
xs = 2*L*cos(w0*t);
ys =   L*sin(w0*t);

% Geometry -> delays
r1 = hypot(xs-mic1(1), ys-mic1(2));
r2 = hypot(xs-mic2(1), ys-mic2(2));

tau1_true = r1/v;
tau2_true = r2/v;
dtau_true = tau2_true - tau1_true;

theta_deg = wrapTo180(rad2deg(atan2(ys,xs)));

% Attenuation gamma=(r0/r)^2, amplitude sqrt(gamma)
g1 = (r0./r1).^2;  a1 = sqrt(g1);
g2 = (r0./r2).^2;  a2 = sqrt(g2);

%% -------------------- 1) Load music signal s(t) --------------------
rng(1);
audioPath = 'Ryan Gosling - City Of Stars (320).mp3';
[s_raw, fs_audio] = audioread(audioPath);

if size(s_raw,2) > 1
    s_raw = mean(s_raw,2);
end
if fs_audio ~= fs
    s_raw = resample(s_raw, fs, fs_audio);
end
if length(s_raw) < Ns
    s_raw(end+1:Ns) = 0;
end
s = s_raw(1:Ns);

% DC removal + normalize
s = s - mean(s);
s = s / std(s);
Ps = var(s); % ~1

%% -------------------- 2) Clean received signals --------------------
s1_clean = apply_timevarying_delay_gain(s, fs, tau1_true, a1);
s2_clean = apply_timevarying_delay_gain(s, fs, tau2_true, a2);

%% -------------------- 3) Framing --------------------
Lw_ms = 100;               
Lw    = round(Lw_ms/1000*fs);
hop   = round(Lw/4);         % 75% overlap
frame_centers = (1+floor(Lw/2)) : hop : (Ns-floor(Lw/2));
B = numel(frame_centers);

maxLag = ceil(1.5*(d/v)*fs);  % ~13 samples
upsamp = 10;                   

fprintf('Window Lw=%d samples (%.1f ms), hop=%d, frames=%d, maxLag=%d\n', ...
    Lw, 1000*Lw/fs, hop, B, maxLag);

dtau_true_frames  = dtau_true(frame_centers);
theta_frames      = theta_deg(frame_centers);

%% -------------------- 4) Monte-Carlo: MSE vs angle (SNRref list) --------------------
SNRref_dB_list = [0 10 20];
MC = 50;

mse_vs_angle = zeros(numel(SNRref_dB_list), B);

for sIdx = 1:numel(SNRref_dB_list)
    SNRref_dB  = SNRref_dB_list(sIdx);
    SNRref_lin = 10^(SNRref_dB/10);

    % Global noise floor from SNRref at r0, using source power Ps
    sigma = sqrt(Ps / SNRref_lin);

    err2_acc = zeros(B,1);

    for mc = 1:MC
        r1_noisy = s1_clean + sigma*randn(Ns,1);
        r2_noisy = s2_clean + sigma*randn(Ns,1);

        dtau_hat_frames = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);

        e = dtau_hat_frames - dtau_true_frames;
        err2_acc = err2_acc + e.^2;
    end

    mse_vs_angle(sIdx,:) = (err2_acc/MC).';
end

%% -------------------- 5) MSE vs angle --------------------
binDeg = 10;                         % 10° bins like your reference style
binEdges   = -180:binDeg:180;
binCenters = (binEdges(1:end-1)+binEdges(2:end))/2;

mse_bin_mean = nan(numel(SNRref_dB_list), numel(binCenters));
mse_bin_std  = nan(numel(SNRref_dB_list), numel(binCenters));

for sIdx = 1:numel(SNRref_dB_list)
    for b = 1:numel(binCenters)
        inBin = theta_frames >= binEdges(b) & theta_frames < binEdges(b+1);
        if any(inBin)
            vals = mse_vs_angle(sIdx,inBin);
            mse_bin_mean(sIdx,b) = mean(vals);
            mse_bin_std(sIdx,b)  = std(vals);
        end
    end
end

figure('Name','Angle + Local SNR (report-style)');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Left: MSE vs angle 
nexttile; hold on; grid on;
for sIdx = 1:numel(SNRref_dB_list)
    semilogy(binCenters, mse_bin_mean(sIdx,:), 'o-', 'LineWidth', 1.5);
end

set(gca,'YScale','log');
xlabel('Source Angle \theta (degrees)');
ylabel('MSE of \Delta\tau\_hat (s^2)');
title('MSE vs Source Angle (Global SNR Model)');
legend(arrayfun(@(x)sprintf('SNR_{ref} = %d dB',x),SNRref_dB_list,'uni',0),'Location','best');

% Right: Local SNR vs angle
nexttile; hold on; grid on;

% Local SNR model: SNR_local(t) = gamma(t) * SNRref (power scaling)
% Use average attenuation across two mics for plotting (smooth, symmetric)
gamma_avg = 0.5*(g1(frame_centers) + g2(frame_centers));

% Bin local SNR for same bins (mean dB)
snr_local_bin = nan(numel(SNRref_dB_list), numel(binCenters));
for sIdx = 1:numel(SNRref_dB_list)
    SNRref_lin = 10^(SNRref_dB_list(sIdx)/10);
    snr_local_lin = gamma_avg * SNRref_lin;
    snr_local_dB  = 10*log10(snr_local_lin);

    for b = 1:numel(binCenters)
        inBin = theta_frames >= binEdges(b) & theta_frames < binEdges(b+1);
        if any(inBin)
            snr_local_bin(sIdx,b) = mean(snr_local_dB(inBin));
        end
    end
end

for sIdx = 1:numel(SNRref_dB_list)
    plot(binCenters, snr_local_bin(sIdx,:), 'o-');
end
xlabel('Source Angle \theta (degrees)');
ylabel('Actual Local SNR (dB)');
title('Local SNR vs Angle (varies due to attenuation)');
legend(arrayfun(@(x)sprintf('SNR_{ref} = %d dB',x),SNRref_dB_list,'uni',0),'Location','best');

%% -------------------- 6) Spatial MSE distribution on ellipse (SNRref = 20 dB) --------------------
doSpatialPlot = true;
if doSpatialPlot
    SNRref_dB = 20;
    SNRref_lin = 10^(SNRref_dB/10);
    sigma = sqrt(Ps / SNRref_lin);

    % Estimate per-frame MSE map (Monte-Carlo averaged)
    err2_acc = zeros(B,1);
    MCmap = 60;

    for mc = 1:MCmap
        r1_noisy = s1_clean + sigma*randn(Ns,1);
        r2_noisy = s2_clean + sigma*randn(Ns,1);
        dtau_hat = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);

        e = dtau_hat - dtau_true_frames;
        err2_acc = err2_acc + e.^2;
    end
    mse_map = err2_acc/MCmap;

    % Sample positions on ellipse at frame centers
    x_f = xs(frame_centers);
    y_f = ys(frame_centers);

    figure('Name','Spatial MSE Distribution (SNRref=20 dB)');
    hold on; grid on; axis equal;

    plot(xs, ys, 'k--', 'LineWidth', 1);        % trajectory
    plot(mic1(1), mic1(2), '^', 'MarkerSize', 10, 'LineWidth', 2);
    plot(mic2(1), mic2(2), '^', 'MarkerSize', 10, 'LineWidth', 2);

    plot(x_f, y_f, 'o', 'MarkerSize', 5, 'LineWidth', 1.2);   % NORMAL plot

    xlabel('x (meters)');
    ylabel('y (meters)');
    title('Spatial MSE Distribution (SNR_{ref} = 20 dB)');
    legend('Trajectory','Mic 1 (Left)','Mic 2 (Right)','MSE samples','Location','best');

end

%% -------------------- 7) MSE vs SNRref at 3 positions + CRB  --------------------
t_picks = [0.5 1.6 3.5];   %

bPick = zeros(size(t_picks));
for i = 1:numel(t_picks)
    [~, bPick(i)] = min(abs(t(frame_centers) - t_picks(i)));
end

SNRref_sweep_dB = -20:5:40;
MC2 = 60;

mse_snr = zeros(numel(bPick), numel(SNRref_sweep_dB));
crb_snr = zeros(numel(bPick), numel(SNRref_sweep_dB));

beta_rms_sq = effective_beta_rms_sq(s, fs);

for p = 1:numel(bPick)
    bIdx = bPick(p);
    n0 = frame_centers(bIdx);

    % Use avg distance for local SNR in CRB
    gamma_local = (r0 / r1(n0))^2;


    for k = 1:numel(SNRref_sweep_dB)
        SNRref_dB  = SNRref_sweep_dB(k);
        SNRref_lin = 10^(SNRref_dB/10);
        sigma = sqrt(Ps / SNRref_lin);

        % Local SNR for CRB
        SNRloc_lin = gamma_local * SNRref_lin;
        crb_snr(p,k) = 1/(Lw * SNRloc_lin * beta_rms_sq);

        err2 = 0;
        for mc = 1:MC2
            r1_noisy = s1_clean + sigma*randn(Ns,1);
            r2_noisy = s2_clean + sigma*randn(Ns,1);

            dtau_hat = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);
            e = dtau_hat(bIdx) - dtau_true_frames(bIdx);
            err2 = err2 + e^2;
        end
        mse_snr(p,k) = err2/MC2;
    end
end

figure('Name','MSE vs SNRref + CRB'); hold on; grid on;

crbStyle = {'--','-.',':'};   % 3 distinct styles

for p = 1:numel(bPick)
    semilogy(SNRref_sweep_dB, mse_snr(p,:), 'o-'); 
    semilogy(SNRref_sweep_dB, crb_snr(p,:), crbStyle{p}, 'LineWidth', 2);
end

xlabel('SNR_{ref} (dB)');
ylabel('MSE (s^2)');
title(sprintf('MSE vs SNR (Upsample x%d) - DToA & CRB', upsamp));

leg = strings(1,2*numel(bPick));
for p = 1:numel(bPick)
    leg(2*p-1) = sprintf('Exp t=%.1fs', t(frame_centers(bPick(p))));
    leg(2*p)   = sprintf('CRB  t=%.1fs', t(frame_centers(bPick(p))));
end
legend(leg,'Location','northeast');
set(gca,'YScale','log');

%% ========================= FUNCTIONS =========================

function y = apply_timevarying_delay_gain(x, fs, tau_t, a_t)
% y(t) = a(t) * x(t - tau(t)) via interpolation
    N  = numel(x);
    tt = (0:N-1)'/fs;
    tq = tt - tau_t(:);
    y  = interp1(tt, x, tq, 'pchip', 0);
    y  = a_t(:).*y;
end

function dtau_hat = estimate_dtau_frames_xcorr(r1, r2, fs, centers, Lw, maxLag, upsamp)

    B = numel(centers);
    dtau_hat = zeros(B,1);
    w = hann(Lw);

    for b = 1:B
        c0 = centers(b);
        idx = (c0-floor(Lw/2)):(c0+ceil(Lw/2)-1);

        f1 = r1(idx).*w;
        f2 = r2(idx).*w;

        [c,lags] = xcorr(f2,f1,maxLag,'coeff');

        % peak on abs correlation
        [~,k0] = max(abs(c));

        if upsamp <= 1
            lag_hat = parabolic_peak(lags, c, k0);
            dtau_hat(b) = lag_hat/fs;
            continue;
        end

        % Upsample correlation in a local window around peak
        win = 6; % samples around peak to upsample 
        kL = max(1, k0-win);
        kR = min(numel(c), k0+win);

        c_loc    = c(kL:kR);
        lags_loc = lags(kL:kR);

        % Build fine lag grid
        lag_fine = linspace(lags_loc(1), lags_loc(end), numel(lags_loc)*upsamp);
        c_fine   = interp1(lags_loc, c_loc, lag_fine, 'pchip');

        % Peak on abs in fine grid
        [~,kf] = max(abs(c_fine));

        % Parabolic refinement on fine grid
        lag_hat = parabolic_peak(lag_fine(:), c_fine(:), kf);

        dtau_hat(b) = lag_hat/fs;
    end
end

function lag_hat = parabolic_peak(lags, c, k)
% Parabolic interpolation around peak index k on arbitrary lag grid.
    if k<=1 || k>=numel(c)
        lag_hat = lags(k); return;
    end
    y1 = c(k-1); y2 = c(k); y3 = c(k+1);
    denom = (y1 - 2*y2 + y3);
    if abs(denom) < 1e-12
        delta = 0;
    else
        delta = 0.5*(y1 - y3)/denom;
    end

  
    lag_step = lags(k+1) - lags(k);
    lag_hat = lags(k) + delta*lag_step;
end

function beta_rms_sq = effective_beta_rms_sq(s, fs)
% beta_rms^2 = (∫ ω^2 |S(ω)|^2 dω)/(∫ |S(ω)|^2 dω) using FFT positive freqs
    S2 = abs(fft(s)).^2;
    L  = numel(S2);
    f  = (0:L-1)'*(fs/L);

    half = 1:floor(L/2);
    fpos = f(half);
    Spos = S2(half);

    omega = 2*pi*fpos;
    beta_rms_sq = sum((omega.^2).*Spos) / max(sum(Spos), eps);
end


%% ===================== (1) dtau_hat vs dtau_true + error(t) =====================
SNRref_one_dB   = 10;          
MC_hist         = 80;          % MC for histogram/CDF
MC_fail         = 60;          % MC for failure rate
SNRref_sweep_dB = -20:5:40;    % sweep for failure rate

fail_thresh_sec = 1/fs;       

t_frames = t(frame_centers);


SNRref_lin = 10^(SNRref_one_dB/10);
sigma = sqrt(Ps / SNRref_lin);

r1_noisy = s1_clean + sigma*randn(Ns,1);
r2_noisy = s2_clean + sigma*randn(Ns,1);

dtau_hat_1 = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);

e1 = dtau_hat_1 - dtau_true_frames;

figure('Name','(1) dtau\_hat vs dtau\_true + error(t)');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t_frames, dtau_true_frames, 'LineWidth', 1.3);
plot(t_frames, dtau_hat_1, '--', 'LineWidth', 1.3);
xlabel('Time (s)'); ylabel('\Delta\tau (s)');
title(sprintf('\\Deltatau true vs \\Deltatau hat (one run), SNR_{ref}=%d dB', SNRref_one_dB));
legend('\Delta\tau_{true}','\Delta\tau_{hat}','Location','best');

nexttile; hold on; grid on;
plot(t_frames, e1, 'LineWidth', 1.2);
yline( fail_thresh_sec, ':', 'LineWidth', 1.2);
yline(-fail_thresh_sec, ':', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Error e(t) (s)');
title(sprintf('Error over time (threshold = %.3g s)', fail_thresh_sec));

%% ===================== (2) Error histogram + CDF (one SNRref) =====================
err_all = [];

for mc = 1:MC_hist
    r1_noisy = s1_clean + sigma*randn(Ns,1);
    r2_noisy = s2_clean + sigma*randn(Ns,1);

    dtau_hat = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);
    err_all = [err_all; (dtau_hat - dtau_true_frames)];
end

abs_err = abs(err_all);

figure('Name','(2) Error histogram + CDF');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
histogram(err_all, 60, 'Normalization','pdf');
xlabel('Error e (s)'); ylabel('PDF');
title(sprintf('Error PDF (MC=%d), SNR_{ref}=%d dB', MC_hist, SNRref_one_dB));

nexttile; hold on; grid on;
[Fa, Xa] = ecdf(abs_err);
plot(Xa, Fa, 'LineWidth', 1.5);
xline(fail_thresh_sec, ':', 'LineWidth', 1.2);
xlabel('|Error| (s)'); ylabel('CDF');
title('CDF of |Error|');
legend('CDF','threshold','Location','best');

%% ===================== (3) Failure rate vs SNRref =====================
fail_rate = zeros(size(SNRref_sweep_dB));

for k = 1:numel(SNRref_sweep_dB)
    SNRref_dB  = SNRref_sweep_dB(k);
    SNRref_lin = 10^(SNRref_dB/10);
    sigma = sqrt(Ps / SNRref_lin);

    fails = 0; total = 0;

    for mc = 1:MC_fail
        r1_noisy = s1_clean + sigma*randn(Ns,1);
        r2_noisy = s2_clean + sigma*randn(Ns,1);

        dtau_hat = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);
        e = dtau_hat - dtau_true_frames;

        bad = abs(e) > fail_thresh_sec;
        fails = fails + sum(bad);
        total = total + numel(bad);
    end

    fail_rate(k) = fails / total;
end

figure('Name','(3) Failure rate vs SNRref'); hold on; grid on;
plot(SNRref_sweep_dB, fail_rate, 'o-', 'LineWidth', 1.5);
xlabel('SNR_{ref} (dB)'); ylabel('Failure rate');
title(sprintf('Failure rate vs SNR_{ref} (MC=%d, threshold=%.3g s)', MC_fail, fail_thresh_sec));
ylim([0 1]);
