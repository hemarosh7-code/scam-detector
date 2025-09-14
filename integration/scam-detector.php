<?php
// wp-content/plugins/scam-detector/scam-detector.php

/**
 * Plugin Name: Scam Detector
 * Description: Integrates scam detection API with WordPress community platform
 * Version: 1.0.0
 */

class ScamDetectorPlugin {
    private $api_url;
    private $api_key;
    
    public function __construct() {
        $this->api_url = get_option('scam_detector_api_url', 'https://api.scamdetector.com');
        $this->api_key = get_option('scam_detector_api_key');
        
        add_action('init', array($this, 'init'));
        add_action('admin_menu', array($this, 'admin_menu'));
        add_action('wp_ajax_analyze_content', array($this, 'ajax_analyze_content'));
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
    }
    
    public function init() {
        // Hook into comment and post submission
        add_action('pre_comment_approved', array($this, 'check_comment'), 10, 2);
        add_action('save_post', array($this, 'check_post'), 10, 3);
    }
    
    public function check_comment($approved, $commentdata) {
        $content = $commentdata['comment_content'];
        $score = $this->analyze_content($content);
        
        if ($score >= 70) {
            // Block high-risk comments
            wp_die('Your comment has been flagged for review.');
        } elseif ($score >= 40) {
            // Hold for moderation
            return 0;
        }
        
        return $approved;
    }
    
    public function check_post($post_id, $post, $update) {
        if ($post->post_type !== 'post' || $update) {
            return;
        }
        
        $content = $post->post_content . ' ' . $post->post_title;
        $score = $this->analyze_content($content);
        
        // Store score as post meta
        update_post_meta($post_id, '_scam_score', $score);
        
        if ($score >= 70) {
            // Change post status to pending
            wp_update_post(array(
                'ID' => $post_id,
                'post_status' => 'pending'
            ));
            
            // Notify administrators
            $this->notify_admin($post, $score);
        }
    }
    
    private function analyze_content($content) {
        $response = wp_remote_post($this->api_url . '/api/analyze', array(
            'headers' => array(
                'Content-Type' => 'application/json',
                'X-API-Key' => $this->api_key
            ),
            'body' => json_encode(array(
                'question' => $content,
                'category' => 'general'
            )),
            'timeout' => 30
        ));
        
        if (is_wp_error($response)) {
            return 0; // Allow content if API fails
        }
        
        $body = wp_remote_retrieve_body($response);
        $data = json_decode($body, true);
        
        return isset($data['scam_score']) ? $data['scam_score'] : 0;
    }
    
    public function ajax_analyze_content() {
        check_ajax_referer('scam_detector_nonce', 'nonce');
        
        $content = sanitize_textarea_field($_POST['content']);
        $score = $this->analyze_content($content);
        
        wp_send_json_success(array(
            'scam_score' => $score,
            'risk_level' => $this->get_risk_level($score)
        ));
    }
    
    private function get_risk_level($score) {
        if ($score >= 70) return 'HIGH';
        if ($score >= 40) return 'MEDIUM';
        return 'LOW';
    }
    
    public function enqueue_scripts() {
        wp_enqueue_script('scam-detector', plugins_url('js/scam-detector.js', __FILE__), array('jquery'), '1.0.0', true);
        wp_localize_script('scam-detector', 'scamDetector', array(
            'ajax_url' => admin_url('admin-ajax.php'),
            'nonce' => wp_create_nonce('scam_detector_nonce')
        ));
    }
    
    public function admin_menu() {
        add_options_page(
            'Scam Detector Settings',
            'Scam Detector',
            'manage_options',
            'scam-detector',
            array($this, 'admin_page')
        );
    }
    
    public function admin_page() {
        if (isset($_POST['submit'])) {
            update_option('scam_detector_api_url', sanitize_url($_POST['api_url']));
            update_option('scam_detector_api_key', sanitize_text_field($_POST['api_key']));
            echo '<div class="notice notice-success"><p>Settings saved!</p></div>';
        }
        
        $api_url = get_option('scam_detector_api_url', 'https://api.scamdetector.com');
        $api_key = get_option('scam_detector_api_key');
        ?>
        <div class="wrap">
            <h1>Scam Detector Settings</h1>
            <form method="post">
                <table class="form-table">
                    <tr>
                        <th scope="row">API URL</th>
                        <td><input type="url" name="api_url" value="<?php echo esc_attr($api_url); ?>" class="regular-text" /></td>
                    </tr>
                    <tr>
                        <th scope="row">API Key</th>
                        <td><input type="password" name="api_key" value="<?php echo esc_attr($api_key); ?>" class="regular-text" /></td>
                    </tr>
                </table>
                <?php submit_button(); ?>
            </form>
        </div>
        <?php
    }
    
    private function notify_admin($post, $score) {
        $subject = 'High-risk content detected: ' . $post->post_title;
        $message = sprintf(
            "A post with high scam score (%d) has been detected:\n\nTitle: %s\nAuthor: %s\nContent: %s\n\nPlease review: %s",
            $score,
            $post->post_title,
            get_the_author_meta('display_name', $post->post_author),
            substr($post->post_content, 0, 200) . '...',
            admin_url('post.php?post=' . $post->ID . '&action=edit')
        );
        
        wp_mail(get_option('admin_email'), $subject, $message);
    }
}

new ScamDetectorPlugin();
?>